# Unreleased

- Add `padding` field to `Pose3` for better compatibility with `bytemuck` and `spirv`.
- Fix various internal implementation details that are incompatible with `spirv`/`naga`.

# v0.1.3

- Add `From`/`Into` conversions between f32 and f64 type variants.

# v0.1.2

- Add optional `rkyv 0.8` support.