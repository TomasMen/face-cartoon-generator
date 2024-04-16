## To run main program for one image:
```$ python -m main <image_path> <data_dir_path>```
e.g:
```$ python -m main .\data\test-set\face-images\level-1\02-22636558997-80ee36b602-o.png .\data\```

## To run test use:
```$ python -m tests.full_test .\data\test-set\ .\data\```

## To run dataset creation set:
```$ python -m scripts.create_fid_cartoons D:\cartoonset10k\cartoonset10k\ D:\````

## To run histogram creation script for version1 test output:
```$ python -m scripts.create_histogram D:\projects\face-cartoon-generator\output\ARCHIVED-TESTS-DNC\v1-testing-results\testing_results.json D:\projects\face-cartoon-generator\output\ARCHIVED-TESTS-DNC\v1-testing-results\```
