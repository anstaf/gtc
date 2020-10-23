#include <gtest/gtest.h>

#include "driver.hpp"
#include "nabla_naive.hpp"

TEST(FVM, nabla) { driver(nabla); }
