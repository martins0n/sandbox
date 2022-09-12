pub fn scalarprod(a: &mut [f32], b: &mut [f32]) -> f32 {
    let sum: f32;
    let n: i32 = a.len() as i32;
    unsafe {
        sum = super::generated::bindings::scalarprod(a.as_mut_ptr(), b.as_mut_ptr(), n);
    }
    sum
}
