# CSE 158 - Core Files
The contents of this repository contain the core files for the UCSD class CSE158 for Fall 2023. I do not claim nor own any of the contents of this repository. It only exists to simplify the assignment process for others. All credits go to Professor Julian McAuley [[contact](mailto:jmcauley@eng.ucsd.edu)]. 
> https://cseweb.ucsd.edu/classes/fa23/cse258-a/

## Usage
1. Fork and clone the repository.
2. Run the unpacker to un-gunzip the `data/` directory. 
```
./unpack.sh
```
3. Change boilerplate to match the filepaths of the `data/` directory. You can also remove the `gzip` stuff inside the boilerplate.

## Notes
- You can skip the unpacker step if you don't want to modify the existing boilerplate. The unpacker just moves the wait from the runtime to the post-install. You will still need to change the filepaths though. 
