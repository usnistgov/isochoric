#ifndef ISOCHOR_DLL_H
#define ISOCHOR_DLL_H

#include <exception>
#include <string>

// Get the platform identifiers, some overlap with "PlatformDetermination.h" from CoolProp's main repo 
#if defined(_WIN32) || defined(__WIN32__) || defined(_WIN64) || defined(__WIN64__)
#    define __ITISWINDOWS__
#endif

#  ifndef EXPORT_CODE
#    if defined(__ITISWINDOWS__)
#      define EXPORT_CODE extern "C" __declspec(dllexport)
#    else
#      define EXPORT_CODE extern "C"
#    endif
#  endif

#  ifndef CONVENTION
#    if defined(__ITISWINDOWS__)
#      define CONVENTION __stdcall
#    else
#      define CONVENTION
#    endif
#  endif

EXPORT_CODE void CONVENTION trace(
    const char *JSON_in,
    double *T, double *p, double *rhoL, double *rhoV, double *x0, double *y0,
    double *errcode, char *JSON_out, const double JSON_out_size);

#endif