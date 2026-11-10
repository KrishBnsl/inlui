import { rmSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";

const projectRoot = resolve(process.cwd());
const cacheDirectory = resolve(projectRoot, ".next");

if (dirname(cacheDirectory) !== projectRoot || basename(cacheDirectory) !== ".next") {
  throw new Error(`Refusing to remove unexpected cache path: ${cacheDirectory}`);
}

rmSync(cacheDirectory, { recursive: true, force: true });
console.log(`Cleared ${cacheDirectory} before mocked E2E compilation.`);
