# CS766-Project-Public

## How to run


- Install Python and packages(scikit-image, numpy, etc..). Or simply install [anaconda](https://www.continuum.io/downloads).

- Clone this repository

```
$ git clone https://github.com/HaiyunJin/CS766_Project.git
$ cd CS766_Project
```

- Compile the Cypthon code (c_seam_carving.pyx)
```
$ python setup.py build_ext --inplace
```

- Run the main algorithm
```
$ python seam_carving.py
```

## OR...
- Compile with
```
$ make
```

- Run with
```
make run
```

- Clean with
```
make clean
```
