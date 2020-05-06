//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void func_a(__global * buffer)
{
	size_t id = get_global_id(0);
    //printf("FUNC A: %d\n", buffer[id]);
    buffer[id] = buffer[id] + id;// * buffer[id];
}

__kernel void func_b(__global * buffer)
{
	size_t id = get_global_id(0);
    //printf("FUNC B: %d\n", buffer[id]);
	buffer[id] = id / buffer[id];// * buffer[id] * buffer[id];
}
__kernel void func_c(__global * buffer)
{
	size_t id = get_global_id(0);
    //printf("FUNC C: %d\n", buffer[id]);
	buffer[id] = buffer[id] * id;// * buffer[id] * buffer[id];
}

__kernel void func_d(__global * buffer)
{
	size_t id = get_global_id(0);
    //printf("FUNC D: %d\n", buffer[id]);
	buffer[id] = buffer[id] ^ id;// * buffer[id] * buffer[id];
}
