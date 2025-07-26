# Note on Large Data Files

The complete Residential_mesh_blocks shapefile contains a large DBF file (215.62 MB) that exceeds GitHub's file size limit. 

## Accessing the Complete Dataset

The full dataset including the large DBF file can be obtained from:

1. **Direct Download**: [Will be provided upon paper publication]
2. **Cloud Storage**: Available on request from the corresponding author
3. **Institutional Repository**: [To be deposited]

## Files Affected

- `Residential_mesh_blocks.dbf` (215.62 MB) - Contains attribute data for 59,483 mesh blocks

## Alternative Access

For analysis purposes, the essential mesh block data is also available in the processed CSV outputs in the `results/` directory, which contain the aggregated statistics used in the paper.

## Using Git LFS

If you need to work with the complete files locally, you can use Git Large File Storage:

```bash
git lfs track "*.dbf"
git add .gitattributes
git add data/Residential_mesh_blocks.dbf
git commit -m "Add large DBF file with Git LFS"
```