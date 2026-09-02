// build.js -- bundles the one source tree in src/ into two self-contained
// dist trees, one per browser target. A single source tree is the point of
// tier 8a: Chrome MV3 wants `background.service_worker`, Firefox needs
// `browser_specific_settings.gecko.id` (required for AMO signing) and
// tolerates the older `background.scripts` array instead of a service
// worker -- see TARGETS below, and demo/README.md for Safari's separate
// Xcode conversion step (not automated: it needs a macOS toolchain this
// build can't assume).
//
// Bundling in webextension-polyfill (rather than branching on typeof
// chrome/browser at runtime) is what lets every module under src/ use
// `browser.*` unconditionally and still work on both targets -- see
// detector-client.js and background.js.
import { build } from "esbuild";
import { mkdirSync, copyFileSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));
const DIST = join(ROOT, "dist");

// Static assets shared byte-for-byte across targets -- only manifest.json
// differs per target (built below from manifest.base.json).
const STATIC_ASSETS = ["popup.html", "overlay.css"];

const TARGETS = {
  chrome: {
    manifest: (base) => ({
      ...base,
      background: { service_worker: "background.js" },
    }),
  },
  firefox: {
    manifest: (base) => ({
      ...base,
      background: { scripts: ["background.js"] },
      browser_specific_settings: {
        gecko: {
          id: "aigc-live-detector@tiktok-techjam-2026",
          strict_min_version: "109.0", // first Firefox release with MV3 support
        },
      },
    }),
  },
};

async function buildTarget(name, spec) {
  const outDir = join(DIST, name);
  rmSync(outDir, { recursive: true, force: true });
  mkdirSync(outDir, { recursive: true });

  // Three independent entry points, bundled together in one call but each
  // producing its own output file: content.js, background.js, and popup.js
  // run in three different execution contexts (page, service
  // worker/background page, popup document) and share nothing at runtime.
  // esbuild just needs to resolve each one's imports (webextension-polyfill
  // plus the other src/*.js modules) into a single flat script MV2/MV3 can
  // load directly -- no <script type="module"> or bundler needed at load
  // time, which matters because Chrome's MV3 service worker doesn't get one
  // unless the manifest opts in, and this manifest deliberately doesn't.
  await build({
    entryPoints: [join(ROOT, "src", "content.js"), join(ROOT, "src", "background.js"), join(ROOT, "src", "popup.js")],
    outdir: outDir,
    bundle: true,
    format: "iife",
    target: "es2020",
    logLevel: "info",
  });

  const base = JSON.parse(readFileSync(join(ROOT, "manifest.base.json"), "utf8"));
  writeFileSync(join(outDir, "manifest.json"), JSON.stringify(spec.manifest(base), null, 2) + "\n");

  for (const asset of STATIC_ASSETS) {
    copyFileSync(join(ROOT, asset), join(outDir, asset));
  }
}

for (const [name, spec] of Object.entries(TARGETS)) {
  await buildTarget(name, spec);
  console.log(`[build] ${name} -> demo/extension/dist/${name}/`);
}
