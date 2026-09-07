# Build history

`design.md` and `plan.md` are the internal design and build records of the migration that created
this repository from the `WIG_Plane_RL_Control` archive in September 2026: the spec the layout was
decided from, and the task-by-task plan that carried it out. They are kept for provenance — they
say why each module sits where it does, which rulings settled the open questions, and what was
checked at each step — and not as user documentation. They are not runnable as written either:
they address a worker with both repositories checked out, use `$OLD` / `$NEW` shell variables, name
absolute build paths, and refer to a `falcon-s-goldens` scratch directory that is not part of the
release. For what this repository does and how to run it, see the top-level `README.md`; for where
its code, data and numbers came from, `PROVENANCE.md`.
