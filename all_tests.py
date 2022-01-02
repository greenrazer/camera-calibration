from test import run_tests

# Since the submodules handle the adding to the test array
# all we need to do is import them
import li_utils_test
import calibrate_camera_test

if __name__ == '__main__':
    run_tests()