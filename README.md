## DATA 5960 FQ 25

This course is an independent study for student who want to contribute data science to problems of social good. 


## Pixi setup and adding necessary packages
To use a jypter notebook in VScode and use a pixi environment, first open the folder in VScode rather than simply opening the individual file from the directory. 

Make sure the pixi.toml and pixi.lock files are both in the same directory as the notebook you want to run. 

Finally, open the notebook from VScode. When you select the kernel, it should be named "default (Python 3.xx.x)" but it its location should be listed as (.pixi/envs/default/bin/python) rather than simply bin/python or usr/bin/python. 

To add packages to the environment, do so in the terminal. Navigate to the directory where you code lives, and run the command `pixi add numpy` or whichever package you need installed. The default source for packages is conda-forge, but if you need a package that is available on pypi (e.g. you would otherwise call `pip install`), you should specify this like `pixi add --pypi numpy`. Only specify pypi if it is strictly necessary (i.e. the package is not available on [conda-forge](https://conda-forge.org/packages/)). 

Please review the documentation for pixi [at the pixi homepage](https://pixi.sh/dev/). 
