// Reads a JSON command on stdin, runs explainer.js, prints JSON. Used by test_demo.py.
const fs = require("fs");
const path = require("path");
const E = require(path.join(__dirname, "..", "explainer.js"));
const P = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "demo_params.json"), "utf8"));
const cmd = JSON.parse(fs.readFileSync(0, "utf8"));
let out;
if (cmd.op === "design") out = cmd.patients.map((p) => E.designRow(p, P.structural));
else if (cmd.op === "explain") out = cmd.cases.map((c) => E.explain(c.patient, c.features, P, c.opts || {}));
else if (cmd.op === "csv") out = E.parseCSV(fs.readFileSync(cmd.file, "utf8")).map((row) =>
  E.explain(cmd.patient, row, P, { rawEcgdeli: true, timing: "after_onset" }));
process.stdout.write(JSON.stringify(out));
