mod generated;
use generated::bindings;
use std::os::unix::ffi::OsStrExt;

const EMBEDDING_SIZE: usize = 768;

fn main() {
    let input_ids = ndarray::arr2(&[[0, 6, 2, 1040, 126, 2508, 127, 2 as i64]]);

    let (batch_size, seq_len) = input_ids.dim();
    let attention_mask =
        ndarray::Array3::ones((batch_size, seq_len, seq_len)).mapv(|x: i32| x == 1);

    let mut env_ptr = std::ptr::null_mut();

    let g_ort = unsafe {
        bindings::OrtGetApiBase().as_ref().unwrap().GetApi.unwrap()(bindings::ORT_API_VERSION)
            .as_ref()
            .unwrap()
    };
    let env_name = std::ffi::CString::new("test").unwrap();
    unsafe {
        g_ort.CreateEnv.unwrap()(
            bindings::OrtLoggingLevel_ORT_LOGGING_LEVEL_FATAL,
            env_name.as_ptr(),
            &mut env_ptr,
        )
    };
    let mut session_options_ptr = std::ptr::null_mut();
    unsafe { g_ort.CreateSessionOptions.unwrap()(&mut session_options_ptr) };
    unsafe { g_ort.SetIntraOpNumThreads.unwrap()(session_options_ptr, 1) };
    assert_ne!(session_options_ptr, std::ptr::null_mut());

    // Sets graph optimization level
    unsafe {
        g_ort.SetSessionGraphOptimizationLevel.unwrap()(
            session_options_ptr,
            bindings::GraphOptimizationLevel_ORT_ENABLE_BASIC,
        )
    };
    let mut session_options_ptr = std::ptr::null_mut();
    unsafe { g_ort.CreateSessionOptions.unwrap()(&mut session_options_ptr) };
    let model_path =
        std::ffi::OsString::from("/Users/marti/Projects/sandbox/codeduplication/tmp/model.onnx");
    let model_path: Vec<std::os::raw::c_char> = model_path
        .as_bytes()
        .iter()
        .chain(std::iter::once(&b'\0'))
        .map(|b| *b as std::os::raw::c_char)
        .collect();

    let mut session_ptr = std::ptr::null_mut();

    unsafe {
        g_ort.CreateSession.unwrap()(
            env_ptr,
            model_path.as_ptr(),
            session_options_ptr,
            &mut session_ptr,
        )
    };

    let mut allocator_ptr = std::ptr::null_mut();

    unsafe { g_ort.GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr) };
    let mut num_input_nodes: u64 = 0;

    let _ = unsafe { g_ort.SessionGetInputCount.unwrap()(session_ptr, &mut num_input_nodes) };
    assert_ne!(num_input_nodes, 0);
    println!("Number of inputs = {:?}", num_input_nodes);

    let mut memory_info_ptr = std::ptr::null_mut();

    unsafe {
        g_ort.CreateCpuMemoryInfo.unwrap()(
            bindings::OrtAllocatorType_OrtArenaAllocator,
            bindings::OrtMemType_OrtMemTypeDefault,
            &mut memory_info_ptr,
        )
    };

    let mut attention_mask_ptr_ptr = std::ptr::null_mut();

    let _ = unsafe {
        g_ort.CreateTensorWithDataAsOrtValue.unwrap()(
            memory_info_ptr,
            attention_mask.as_ptr() as *mut std::os::raw::c_void,
            ((attention_mask.len() as usize) * std::mem::size_of::<bool>()) as u64,
            attention_mask.shape().as_ptr() as *const i64,
            attention_mask.shape().len() as u64,
            bindings::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
            &mut attention_mask_ptr_ptr,
        )
    };

    let mut input_ids_ptr_ptr = std::ptr::null_mut();

    unsafe {
        g_ort.CreateTensorWithDataAsOrtValue.unwrap()(
            memory_info_ptr,
            input_ids.as_ptr() as *mut std::os::raw::c_void,
            ((input_ids.len() as usize) * std::mem::size_of::<i64>()) as u64,
            input_ids.shape().as_ptr() as *const i64,
            input_ids.shape().len() as u64,
            bindings::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &mut input_ids_ptr_ptr,
        )
    };

    let input_node_names_ptr_ptr = ["input_ids", "attention_mask"]
        .map(|n| std::ffi::CString::new(n.clone()).unwrap().into_raw() as *const i8)
        .as_ptr();
    let output_node_names_ptr_ptr = ["last_hidden_state"]
        .map(|n| std::ffi::CString::new(n.clone()).unwrap().into_raw() as *const i8)
        .as_ptr();

    let run_options_ptr: *const bindings::OrtRunOptions = std::ptr::null();

    let mut output_tensor_ptr: *mut bindings::OrtValue = std::ptr::null_mut();
    let output_tensor_ptr_ptr: *mut *mut bindings::OrtValue = &mut output_tensor_ptr;
    println!("Number of inputs = {:?}", num_input_nodes);

    let input = [
        input_ids_ptr_ptr as *const bindings::OrtValue,
        attention_mask_ptr_ptr as *const bindings::OrtValue,
    ]
    .as_ptr();

    unsafe {
        g_ort.Run.unwrap()(
            session_ptr,
            run_options_ptr,
            input_node_names_ptr_ptr,
            input,
            2,
            output_node_names_ptr_ptr,
            1,
            output_tensor_ptr_ptr,
        )
    };

    assert_ne!(output_tensor_ptr, std::ptr::null_mut());

    let mut output: *mut f32 = std::ptr::null_mut();
    let output_ptr: *mut *mut f32 = &mut output;
    let output_ptr_void: *mut *mut std::ffi::c_void = output_ptr as *mut *mut std::ffi::c_void;

    unsafe {
        g_ort.GetTensorMutableData.unwrap()(output_tensor_ptr, output_ptr_void);
    }

    let token_embeddings =
        unsafe { std::slice::from_raw_parts(output, batch_size * seq_len * EMBEDDING_SIZE) };

    token_embeddings.iter().for_each(|v| println!("{:?}", v));

    let sum = token_embeddings.iter().sum::<f32>() / seq_len as f32;

    println!("Sum {}", sum);
    assert!((sum - 23.19685935229063).abs() < 0.0001);
}
