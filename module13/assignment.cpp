//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "info.hpp"
#include "performance_helper.h"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

#define NUM_BUFFER_ELEMENTS 16

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

bool load_values( const std::string& input_file, std::vector<int>& values )
{
    std::ifstream input(input_file);
    if ( !input.is_open() )
        return false;

    std::string line;
    std::stringstream ss(line);
    std::stringstream valueStream;
    int input_val;
    while( std::getline(input, line) )
    {  
        ss << line;
        std::string value;
        
        while(std::getline(ss, value, ','))
        {   
            valueStream.str(value);
            valueStream.clear();
            valueStream >> input_val;
            values.push_back(input_val);
        }
        ss.str(std::string());
        ss.clear();
    }
    return true;
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context0;
    cl_program program0;
    int * inputOutput0;
    cl_context context1;
    cl_program program1;
    int * inputOutput1;
    cl_context context2;
    cl_program program2;
    int * inputOutput2;
    cl_context context3;
    cl_program program3;
    int * inputOutput3;

    std::vector<int> input;
    int platform = DEFAULT_PLATFORM; 

    if ( argc <= 1 )
    {
        std::cout << "Please provide an input file\n";
        exit(1);
    }
    
    std::string file_name = argv[1];
    std::cout << "FILE NAME: " << file_name << "\n";
    if ( !load_values(file_name, input) )
    {
        std::cout << "Failed to read input file\n";
        exit(1);
    }
    
    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

   // std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    std::ifstream srcFile("assignment.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
/*    DisplayPlatformInfo(
        platformIDs[platform], 
        CL_PLATFORM_VENDOR, 
        "CL_PLATFORM_VENDOR");
*/
    errNum = clGetDeviceIDs(
        platformIDs[platform], 
        CL_DEVICE_TYPE_ALL, 
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }       

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        numDevices, 
        &deviceIDs[0], 
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };

    context0 = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

    context1 = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");
 
    context2 = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

    context3 = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");
 
    // Create program from source
    program0 = clCreateProgramWithSource(
        context0, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

     // Create program from source
    program1 = clCreateProgramWithSource(
        context1, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");
  
    program2 = clCreateProgramWithSource(
        context2, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");
 
    program3 = clCreateProgramWithSource(
        context3, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");
 
 
    // Build program
    errNum = clBuildProgram(
        program0,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
 
     // Build program
    errNum |= clBuildProgram(
        program1,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);

    errNum |= clBuildProgram(
        program2,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
 
    errNum |= clBuildProgram(
        program3,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
 
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog0[16384];
        clGetProgramBuildInfo(
            program0, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog0), 
            buildLog0, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog0;
            checkErr(errNum, "clBuildProgram");
     
        // Determine the reason for the error
        char buildLog1[16384];
        clGetProgramBuildInfo(
            program1, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog1), 
            buildLog1, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog1;
            checkErr(errNum, "clBuildProgram");
        
        char buildLog2[16384];
        clGetProgramBuildInfo(
            program2, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog2), 
            buildLog1, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog2;
            checkErr(errNum, "clBuildProgram");

        char buildLog3[16384];
        clGetProgramBuildInfo(
            program3, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog1), 
            buildLog3, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog3;
            checkErr(errNum, "clBuildProgram");
    }

    std::vector<int> mod3Buf, mod4Buf, mod5Buf, otherBuf;

    for ( int i : input )
    {
        if ( i % 3 == 0 )
        {
            mod3Buf.push_back(i);
        }
        else if ( i % 4 == 0 )
        {
            mod4Buf.push_back(i);
        }
        else if ( i % 5 == 0 )
        {
            mod5Buf.push_back(i);
        }
        else
        {
            otherBuf.push_back(i);
        }
    }
    int i0_len = mod3Buf.size();
    int i1_len = mod4Buf.size();
    int i2_len = mod5Buf.size();
    int i3_len = otherBuf.size();

    // create buffers and sub-buffers
    inputOutput0 = new int[i0_len * numDevices];
    for (unsigned int i = 0; i < i0_len * numDevices; i++)
    {
        inputOutput0[i] = mod3Buf[i];
    }
 
    inputOutput1 = new int[i1_len * numDevices];
    for (unsigned int i = 0; i < i1_len * numDevices; i++)
    {
        inputOutput1[i] = mod4Buf[i];
    }

    inputOutput2 = new int[i2_len * numDevices];
    for (unsigned int i = 0; i < i2_len * numDevices; i++)
    {
        inputOutput2[i] = mod5Buf[i];
    }
 
    inputOutput3 = new int[i3_len * numDevices];
    for (unsigned int i = 0; i < i3_len * numDevices; i++)
    {
        inputOutput3[i] = otherBuf[i];
    }

    // create a single buffer to cover all the input data
    cl_mem buffer0 = clCreateBuffer(
        context0,
        CL_MEM_READ_WRITE,
        sizeof(int) * i0_len * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");
 
     // create a single buffer to cover all the input data
    cl_mem buffer1 = clCreateBuffer(
        context1,
        CL_MEM_READ_WRITE,
        sizeof(int) * i1_len * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    cl_mem buffer2 = clCreateBuffer(
        context2,
        CL_MEM_READ_WRITE,
        sizeof(int) * i2_len * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");
 
    cl_mem buffer3 = clCreateBuffer(
        context3,
        CL_MEM_READ_WRITE,
        sizeof(int) * i3_len * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    // Create command queues
    /*InfoDevice<cl_device_type>::display(
     	deviceIDs[0], 
     	CL_DEVICE_TYPE, 
     	"CL_DEVICE_TYPE");
*/
    cl_command_queue queue0 = 
     	clCreateCommandQueue(
     	context0,
     	deviceIDs[0],
     	0,
     	&errNum);
    checkErr(errNum, "clCreateCommandQueue");
 
    cl_command_queue queue1 = 
     	clCreateCommandQueue(
     	context1,
     	deviceIDs[0],
     	0,
     	&errNum);
    checkErr(errNum, "clCreateCommandQueue");
 
    cl_command_queue queue2 = 
     	clCreateCommandQueue(
     	context2,
     	deviceIDs[0],
     	0,
     	&errNum);
    checkErr(errNum, "clCreateCommandQueue");
 
    cl_command_queue queue3 = 
     	clCreateCommandQueue(
     	context3,
     	deviceIDs[0],
     	0,
     	&errNum);
    checkErr(errNum, "clCreateCommandQueue");
 
    cl_kernel kernel0 = clCreateKernel(
     program0,
     "func_a",
     &errNum);
    checkErr(errNum, "clCreateKernel(func_a)");
 
    cl_kernel kernel1 = clCreateKernel(
     program1,
     "func_b",
     &errNum);
    checkErr(errNum, "clCreateKernel(func_b)");
 
    cl_kernel kernel2 = clCreateKernel(
     program2,
     "func_c",
     &errNum);
    checkErr(errNum, "clCreateKernel(func_c)");
 
    cl_kernel kernel3 = clCreateKernel(
     program3,
     "func_d",
     &errNum);
    checkErr(errNum, "clCreateKernel(func_d)");

    errNum = clSetKernelArg(kernel0, 0, sizeof(cl_mem), (void *)&buffer0);
    checkErr(errNum, "clSetKernelArg(func_a)");

    errNum = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&buffer1);
    checkErr(errNum, "clSetKernelArg(func_b)");
 
    errNum = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&buffer2);
    checkErr(errNum, "clSetKernelArg(func_c)");

    errNum = clSetKernelArg(kernel3, 0, sizeof(cl_mem), (void *)&buffer3);
    checkErr(errNum, "clSetKernelArg(func_d)");
 
    // Write input data
    errNum = clEnqueueWriteBuffer(
      queue0,
      buffer0,
      CL_TRUE,
      0,
      sizeof(int) * i0_len * numDevices,
      (void*)inputOutput0,
      0,
      NULL,
      NULL);
 
    errNum = clEnqueueWriteBuffer(
      queue1,
      buffer1,
      CL_TRUE,
      0,
      sizeof(int) * i1_len * numDevices,
      (void*)inputOutput1,
      0,
      NULL,
      NULL);
  
    // Write input data
    errNum = clEnqueueWriteBuffer(
      queue2,
      buffer2,
      CL_TRUE,
      0,
      sizeof(int) * i2_len * numDevices,
      (void*)inputOutput2,
      0,
      NULL,
      NULL);
 
    errNum = clEnqueueWriteBuffer(
      queue3,
      buffer3,
      CL_TRUE,
      0,
      sizeof(int) * i3_len * numDevices,
      (void*)inputOutput3,
      0,
      NULL,
      NULL);
 
    std::vector<cl_event> events;
    // call kernel for each device
    cl_event event0;

//    size_t gWI = NUM_BUFFER_ELEMENTS;

    size_t gWI0(i0_len);
    size_t gWI1(i1_len);
    size_t gWI2(i2_len);
    size_t gWI3(i3_len);
    errNum = clEnqueueNDRangeKernel(
      queue0, 
      kernel0, 
      1, 
      NULL,
      (const size_t*)&gWI0, 
      (const size_t*)NULL, 
      0, 
      0, 
      &event0);
	
 	cl_event event1;

    errNum = clEnqueueNDRangeKernel(
      queue1, 
      kernel1, 
      1, 
      NULL,
      (const size_t*)&gWI1, 
      (const size_t*)NULL, 
      0, 
      0, 
      &event1); 
 
 	cl_event event2;
 	errNum = clEnqueueMarker(queue2, &event2);
    errNum = clEnqueueNDRangeKernel(
      queue2, 
      kernel2, 
      1, 
      NULL,
      (const size_t*)&gWI2, 
      (const size_t*)NULL, 
      0, 
      0, 
      &event2);
	
 	cl_event event3;
 	errNum = clEnqueueMarker(queue3, &event3);

    errNum = clEnqueueNDRangeKernel(
      queue3, 
      kernel3, 
      1, 
      NULL,
      (const size_t*)&gWI3, 
      (const size_t*)NULL, 
      0, 
      0, 
      &event3); 
 	
 	//Wait for queue 1 to complete before continuing on queue 0
    auto start = get_clock_time();
    errNum = clEnqueueBarrier(queue0);
    auto stop = get_clock_time();
    auto queue0_duration = get_duration_ns(start, stop);
    
    start = get_clock_time();
    errNum = clEnqueueWaitForEvents(queue0, 1, &event1);
    stop = get_clock_time();
    auto queue1_duration = get_duration_ns(start, stop);

    start = get_clock_time();
 	errNum = clEnqueueWaitForEvents(queue1, 1, &event2);    
    stop = get_clock_time();
    auto queue2_duration = get_duration_ns(start, stop);

    start = get_clock_time();
 	errNum = clEnqueueWaitForEvents(queue2, 1, &event3);
    stop = get_clock_time();
    auto queue3_duration = get_duration_ns(start, stop);
	
    // Read back computed data
   	clEnqueueReadBuffer(
            queue0,
            buffer0,
            CL_TRUE,
            0,
            sizeof(int) * i0_len * numDevices,
            (void*)inputOutput0,
            0,
            NULL,
            NULL);
   	clEnqueueReadBuffer(
            queue1,
            buffer1,
            CL_TRUE,
            0,
            sizeof(int) * i1_len * numDevices,
            (void*)inputOutput1,
            0,
            NULL,
            NULL);
 
    clEnqueueReadBuffer(
            queue2,
            buffer2,
            CL_TRUE,
            0,
            sizeof(int) * i2_len * numDevices,
            (void*)inputOutput2,
            0,
            NULL,
            NULL);
   	clEnqueueReadBuffer(
            queue3,
            buffer3,
            CL_TRUE,
            0,
            sizeof(int) * i3_len * numDevices,
            (void*)inputOutput3,
            0,
            NULL,
            NULL);
/* 
    // Display output in rows
    for (unsigned elems = 0; elems < i0_len; elems++)
    {
     std::cout << " " << inputOutput0[elems];
    }
    std::cout << std::endl;
 
    for (unsigned elems = 0; elems < i1_len; elems++)
    {
     std::cout << " " << inputOutput1[elems];
    }
    std::cout << std::endl;
 
    for (unsigned elems = 0; elems < i2_len; elems++)
    {
     std::cout << " " << inputOutput2[elems];
    }
    std::cout << std::endl;
 
    for (unsigned elems = 0; elems < i3_len; elems++)
    {
     std::cout << " " << inputOutput3[elems];
    }
    std::cout << std::endl;
 
    std::cout << "Program completed successfully" << std::endl;
*/
    std::cout << "Queue 0 took " << queue0_duration << " ns.\n";
    std::cout << "Queue 1 took " << queue1_duration << " ns.\n";
    std::cout << "Queue 2 took " << queue2_duration << " ns.\n";
    std::cout << "Queue 3 took " << queue3_duration << " ns.\n";
 
    return 0;
}
