# Paint and save multiscale labels

Run the Lodstone example with a local label store.

```sh
uv run --extra zarr --with matplotlib examples/label_pyramid.py --labels labels.zarr
```

Left-drag paints label 1 and right-drag erases. Ctrl+Z and Ctrl+Y undo and redo strokes. Ctrl+S saves to the store. Run the same command to reopen it. The Matplotlib toolbar supplies pan and zoom. Exit the toolbar's navigation mode to return to painting.

`ZarrLabels.create(path, reference.pyramid)` creates sparse unsigned-integer arrays aligned to the reference. `LabelEditor` supplies asynchronous region reads and a world-space `stroke` method. Each stroke affects one finest-grid voxel along every fixed axis. The brush radius is in physical units on the plane. The display level has no effect on voxel selection.

The writable pyramid supports aligned integer downsampling factors with a common origin. Each derived voxel takes the finest-grid label containing its center. An exact boundary selects the higher index, and a center beyond a partial edge uses the last finest voxel. Background participates like any other ID and thin structures can disappear at coarse levels. Undo and redo apply in memory across saves within the current editing session.

Unsaved edits appear through the editor at every level. Each saved chunk updates its affected derived chunks from the finest array. A local marker lets reopen finish regenerating derived chunks if a process exits during a write. Explicit save can persist a subset of the dirty chunks before an error, and retry saves the remainder. Use one writer per local store.
