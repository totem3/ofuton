#![allow(dead_code)]
extern crate rustfft;
extern crate ndarray;

use rustfft::{FFTnum, FFTplanner};
use rustfft::num_complex::Complex;
use rustfft::num_traits::{Zero};
use ndarray::{ArrayViewMut, ArrayViewMut2, Dimension};

fn _fft<T: FFTnum>(input: &mut [Complex<T>], output: &mut [Complex<T>], inverse: bool) {
    let mut planner = FFTplanner::new(inverse);
    let len = input.len();
    let fft = planner.plan_fft(len);
    fft.process(input, output);
}

fn fft<T: FFTnum>(input: &mut [Complex<T>], output: &mut [Complex<T>]) {
    _fft(input, output, false);
}

fn ifft<T: FFTnum + From<u32>>(input: &mut [Complex<T>], output: &mut [Complex<T>]) {
    _fft(input, output, true);
    for v in output.iter_mut() {
        *v = v.unscale(T::from(input.len() as u32));
    }
}

fn fft2(input: &mut ArrayViewMut2<Complex<f64>>, output: &mut ArrayViewMut2<Complex<f64>>) {
    fftnd(input, output, &[0,1]);
}

fn ifft2(input: &mut ArrayViewMut2<Complex<f64>>, output: &mut ArrayViewMut2<Complex<f64>>) {
    ifftnd(input, output, &[1,0]);
}

fn fftn<D: Dimension>(input: &mut ArrayViewMut<Complex<f64>, D>, output: &mut ArrayViewMut<Complex<f64>, D>, axis: usize) {
    _fftn(input, output, axis, false);
}

fn ifftn<D: Dimension>(input: &mut ArrayViewMut<Complex<f64>, D>, output: &mut ArrayViewMut<Complex<f64>, D>, axis: usize) {
    _fftn(input, output, axis, true);
}

fn _fftn<D: Dimension>(input: &mut ArrayViewMut<Complex<f64>, D>, output: &mut ArrayViewMut<Complex<f64>, D>, axis: usize, inverse: bool) {
    if inverse {
        mutate_lane(input, output, ifft, axis)
    } else {
        mutate_lane(input, output, fft, axis)
    }
}

fn fftnd<D: Dimension>(input: &mut ArrayViewMut<Complex<f64>, D>, output: &mut ArrayViewMut<Complex<f64>, D>, axes: &[usize]) {
    _fftnd(input, output, axes, false);
}

fn ifftnd<D: Dimension>(input: &mut ArrayViewMut<Complex<f64>, D>, output: &mut ArrayViewMut<Complex<f64>, D>, axes: &[usize]) {
    _fftnd(input, output, axes, true);
}

fn _fftnd<D: Dimension>(input: &mut ArrayViewMut<Complex<f64>, D>, output: &mut ArrayViewMut<Complex<f64>, D>, axes: &[usize], inverse: bool) {
    let len = axes.len();
    for i in 0..len {
        let axis = axes[i];
        _fftn(input, output, axis, inverse);
        if i < len - 1 {
            let mut outrows = output.genrows_mut().into_iter();
            for mut row in input.genrows_mut() {
                let mut outrow = outrows.next().unwrap();
                row.as_slice_mut().unwrap().copy_from_slice(outrow.as_slice_mut().unwrap());
            }
        }
    }
}

fn mutate_lane<T: Zero + Clone, D: Dimension>(input: &mut ArrayViewMut<T, D>, output: &mut ArrayViewMut<T, D>, f: fn(&mut [T], &mut [T]) -> (), axis: usize) {
    if axis > 0 {
        input.swap_axes(0, axis);
        output.swap_axes(0, axis);
        {
            let mut outrows = output.genrows_mut().into_iter();
            for mut row in input.genrows_mut() {
                let mut outrow = outrows.next().unwrap();
                let mut vec = row.to_vec();
                let mut out = vec![Zero::zero(); outrow.len()];
                f(&mut vec, &mut out);
                for i in 0..outrow.len() {
                    outrow[i] = out.remove(0);
                }
            }
        }
        input.swap_axes(0, axis);
        output.swap_axes(0, axis);
    } else {
        let mut outrows = output.genrows_mut().into_iter();
        for mut row in input.genrows_mut() {
            let mut outrow = outrows.next().unwrap();
            f(&mut row.as_slice_mut().unwrap(), &mut outrow.as_slice_mut().unwrap());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{fft, ifft, fft2, ifft2};
    use rustfft::num_complex::Complex;
    use rustfft::num_traits::Zero;
    use ndarray::ArrayViewMut;

    fn assert_eq_vecs(a: &[Complex<f64>], b: &[Complex<f64>]) {
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).norm() < 0.1f64);
        }
    }

    #[test]
    fn test_fft() {
        let mut input: Vec<Complex<f64>> = vec![1.,2.,3.,4.,5.,6.,7.,8.,9.].into_iter().map(|x| Complex::new(x, 0.)).collect();
        let mut output = vec![Zero::zero(); 9];
        fft(&mut input, &mut output);
        let expected = [Complex::new(45.0,  0.        ), Complex::new(-4.5, 12.36364839), Complex::new(-4.5,   5.36289117),
                        Complex::new(-4.5,  2.59807621), Complex::new(-4.5,  0.79347141), Complex::new(-4.5,  -0.79347141),
                        Complex::new(-4.5, -2.59807621), Complex::new(-4.5, -5.36289117), Complex::new(-4.5, -12.36364839)];
        assert_eq_vecs(&expected, &output);
    }

    #[test]
    fn test_inverse_fft() {
        let mut input: Vec<Complex<f64>> = vec![1.,2.,3.,4.,5.,6.,7.,8.,9.].into_iter().map(|x| Complex::new(x, 0.)).collect();
        let expected = input.clone();
        let mut output = vec![Zero::zero(); 9];
        fft(&mut input, &mut output);
        let mut output2 = vec![Zero::zero(); 9];
        ifft(&mut output, &mut output2);
        assert_eq_vecs(&expected, &output2);
    }

    #[test]
    fn test_fft2() {
        let mut input: Vec<Complex<f64>> = vec![1.,2.,3.,4.,5.,6.,7.,8.,9.].into_iter().map(|x| Complex::new(x, 0.)).collect();
        let mut input_view = ArrayViewMut::from_shape((3,3), &mut input).unwrap();
        let mut output = vec![Zero::zero(); 9];
        {
            let mut output_view = ArrayViewMut::from_shape((3,3), &mut output).unwrap();
            fft2(&mut input_view, &mut output_view);
        }

        let expected = [Complex::new( 45.0,  0.        ), Complex::new(-4.5, 2.59807621), Complex::new(-4.5, -2.59807621),
                        Complex::new(-13.5,  7.79422863), Complex::new( 0.0, 0.        ), Complex::new( 0.0,  0.        ),
                        Complex::new(-13.5, -7.79422863), Complex::new( 0.0, 0.        ), Complex::new( 0.0,  0.        )];
        assert_eq_vecs(&expected, &output);
    }

    #[test]
    fn test_inverse_fft2() {
        let mut input: Vec<Complex<f64>> = vec![1.,2.,3.,4.,5.,6.,7.,8.,9.].into_iter().map(|x| Complex::new(x, 0.)).collect();
        let mut input_view = ArrayViewMut::from_shape((3,3), &mut input).unwrap();
        let mut output = vec![Zero::zero(); 9];
        let mut output_view = ArrayViewMut::from_shape((3,3), &mut output).unwrap();
        fft2(&mut input_view, &mut output_view);
        let mut output2 = vec![Zero::zero(); 9];
        {
            let mut output2_view = ArrayViewMut::from_shape((3,3), &mut output2).unwrap();
            ifft2(&mut output_view, &mut output2_view);
        }

        let expected: Vec<Complex<f64>> = vec![1.,2.,3.,4.,5.,6.,7.,8.,9.].into_iter().map(|x| Complex::new(x, 0.)).collect();
        assert_eq_vecs(&expected, &output2);
    }
}
