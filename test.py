tests = []

def test(func):
    def wrapper():
        print(f"Starting test \"{func.__name__}\"...")
        passed = True
        try:
            temp_passed = func()
            if temp_passed is not None:
                passed = temp_passed
        except AssertionError as e:
            print(" L x Test Failed with error : ", e)
        else:
            if passed:
                print(" L o Test Passed!")
            else:
                print(" L x Test Failed.")
    tests.append(wrapper)
    return wrapper

def run_tests():
    for test in tests:
        test()