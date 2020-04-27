//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// assignment.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void average(__global float * buffer)
{
	size_t id = get_global_id(0);
    printf("GLOBAL ID: %lu, VAL: %f\n",id, buffer[id]);
    buffer[id] = ( buffer[0] + buffer[1] + buffer[2] + buffer[3] ) / 4;
}
