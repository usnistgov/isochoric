
#define REFPROP_IMPLEMENTATION
// This next definition makes the symbols exposed
#define REFPROP_FUNCTION_MODIFIER 
#define REFPROP_LIB_NAMESPACE REFPROP_lib
#include "REFPROP_lib.h"
#undef REFPROP_LIB_NAMESPACE
#undef REFPROP_FUNCTION_MODIFIER
#undef REFPROP_IMPLEMENTATION

bool my_load_REFPROP(std::string &err, const std::string &shared_library_path = "", const std::string &shared_library_name = "") {
    return REFPROP_lib::load_REFPROP(err, shared_library_path, shared_library_name);
}
bool is_hooked() {
    return REFPROP_lib::SETUPdll != nullptr;
}

