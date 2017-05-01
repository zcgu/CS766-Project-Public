
all:
	python setup.py build_ext --inplace

run:
	python seam_carving.py

clean:
	rm -fr c_seam_carving.so seam_carving_backup.pyc build c_seam_carving.c
