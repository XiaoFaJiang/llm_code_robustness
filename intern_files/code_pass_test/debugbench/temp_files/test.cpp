
#include <iostream>
#include <cassert>

bool is_woodall(long long x) {
    if (x % 2 == 0) {
        return false;
    }
    if (x == 1) {
        return true;
    }
    x = x + 1;
    long long p = 0;
    while (x % 2 == 0) {
        x = x / 2;
        p = p + 1;
        if (p == x) {
            return true;
        }
    }
    return false;
}



int main() {
    assert(is_woodall(383) == true);
    assert(is_woodall(254) == false);
    assert(is_woodall(200) == false);

    assert(is_woodall(32212254719) == true);
    assert(is_woodall(32212254718) == false);
    assert(is_woodall(159) == true);
    
    std::cout << "All tests passed successfully." << std::endl;
    return 0;
}