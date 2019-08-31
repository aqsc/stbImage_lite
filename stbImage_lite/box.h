#pragma once

#ifndef BOX_H
#define BOX_H
#include "darknet.h"

typedef struct {
	float dx, dy, dw, dh;
} dbox;

#endif