#include <gtest/gtest.h>

#include "driver.hpp"
#include "nabla_cuda.hpp"

TEST(FVM, nabla) { driver(nabla); }
