# If you merge the pull request you should either fix all the problems or not delete the branch

# General notes
- there is no `Makefile` nor instruction for compiling of any kind
- You lack a readme file that explains briefly the aim of your project, how to compile and what to expect
- `.vscode` and `.DS_Store` folder should be in the gitignore
- You should not upload the compiled executable but provide instructions 
- You should not put the source files of muparser in the main folder

## Minor
- It would be better is muparser was included as a submodule

# Code
## Major
- You should not include `.cpp` files
- The `Montecarlo_integration` should work for different geometries (at least hyper-rectangles and hyper-balls)
- There is not much code
- Power with integer exponent should not use `std::pow`

## Minor
- use the <number> module for \pi instead of M_PI
- should use std:: for math functions