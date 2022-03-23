# Davis Splits
## The Davis dataset was obtained by using the opensource [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose) repository ( the library has the helper function to download and load the dataset ) 
The files to datasets and folds created using the Davis dataset are present in [data](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/kanakala_ganesh_research_iiit_ac_in/Eh_Q5Gh4A2lBr1aHYTp4eU8BMjBNY-ItiBd91bFOez9R8w?e=Yy4qlm)

The folder contains pickle files for 9 folds, represented as foldxy.pkl, where x,y are protein fold id and ligand fold id respectively, foldxy.pkl is created by crossing all protein in fold x and all ligands in fold y.
Combining three foldsxy.pkl with same x will lead to a fold with 40% similarity threshold.
