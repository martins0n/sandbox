mod generated;
mod scalarprod;

fn main() {
    let mut a: [f32; 3] = [1.0, 2.0, 3.0];
    let mut b: [f32; 3] = [1.0, 2.0, 3.0];
    let sum = scalarprod::scalarprod(&mut a, &mut b);
    println!("sum = {}", sum);
}
