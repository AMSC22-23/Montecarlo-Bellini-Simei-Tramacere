# General notes
- there is no `Makefile` nor instruction for compiling of any kind
- You lack a readme file that explains briefly the aim of your project, how to compile and what to expect
- `.vscode` folder should be in the gitignore
- You should not upload the compiled executable but provide instructions 
- You should not put the source files of muparser in the main folder

## Minor
- It would be better is muparser was included as a submodule

# Code
## Major
- You should not include `.cpp` files
- The declaration of `HyperSphere` should be in a `.hpp` file
- The semantics of the function `generate_random_point` is a bit unclear
- The `Montecarlo_integration` should work for different geometries (at least hyper-rectangles and hyper-balls)
- There is not much code

## Minor
- the number header for \pi instead of M_PI
- should use std:: for math functions