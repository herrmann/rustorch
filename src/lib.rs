#![feature(stdsimd, stmt_expr_attributes)]

use rand::thread_rng;
use rand_distr::{Distribution, Normal, NormalError};
use std::arch::x86_64::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::iter::zip;
use std::rc::Rc;

pub mod tensor;

// Structures

pub enum Op {
    Add,
    Mul,
    Tanh,
    Exp,
    Pow(f32),
}

impl Op {
    pub fn symbol(&self) -> String {
        match self {
            Op::Add => "+".to_string(),
            Op::Mul => "*".to_string(),
            Op::Tanh => "tanh".to_string(),
            Op::Exp => "exp".to_string(),
            Op::Pow(n) => format!("**{}", n),
        }
    }
}

pub struct Value {
    pub data: f32,
    grad: f32,
    label: String,
    prev: Vec<ValueCell>,
    op: Option<Op>,
}

impl Value {
    pub fn new(data: f32, label: String) -> Self {
        Value {
            data,
            grad: 0.,
            label,
            prev: vec![],
            op: None,
        }
    }
}

type ValueCell = Rc<RefCell<Value>>;

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data={})", self.data)
    }
}

// Literals

pub fn lit(data: f32, label: String) -> ValueCell {
    Rc::new(RefCell::new(Value::new(data, label)))
}

// Primitive builders

fn _binary_op(e1: &ValueCell, e2: &ValueCell, op: Op, data: f32) -> ValueCell {
    Rc::new(RefCell::new(Value {
        data,
        grad: 0.,
        label: format!(
            "{} {} {}",
            e1.borrow().label,
            e2.borrow().label,
            op.symbol()
        ),
        prev: vec![Rc::clone(e1), Rc::clone(e2)],
        op: Some(op),
    }))
}

fn _unary_op(e: &ValueCell, op: Op, data: f32) -> ValueCell {
    let v = e.borrow();
    Rc::new(RefCell::new(Value {
        data,
        grad: 0.,
        label: format!("{} {}", v.label, op.symbol()),
        prev: vec![Rc::clone(e)],
        op: Some(op),
    }))
}

// Primitive ops

pub fn add(e1: &ValueCell, e2: &ValueCell) -> ValueCell {
    _binary_op(e1, e2, Op::Add, e1.borrow().data + e2.borrow().data)
}

pub fn mul(e1: &ValueCell, e2: &ValueCell) -> ValueCell {
    _binary_op(e1, e2, Op::Mul, e1.borrow().data * e2.borrow().data)
}

pub fn tanh(e: &ValueCell) -> ValueCell {
    _unary_op(e, Op::Tanh, e.borrow().data.tanh())
}

pub fn exp(e: &ValueCell) -> ValueCell {
    _unary_op(e, Op::Exp, e.borrow().data.exp())
}

pub fn pow(e1: &ValueCell, e2: f32) -> ValueCell {
    _unary_op(e1, Op::Pow(e2), e1.borrow().data.powf(e2))
}

// Auxiliary ops

pub fn sub(e1: &ValueCell, e2: &ValueCell) -> ValueCell {
    add(e1, &neg(e2))
}

pub fn div(e1: &ValueCell, e2: &ValueCell) -> ValueCell {
    mul(e1, &pow(e2, -1.))
}

pub fn neg(e: &ValueCell) -> ValueCell {
    mul(e, &lit(-1., "-1".to_string()))
}

// Backprop

fn build_topo(e: &ValueCell, visited: &mut HashSet<*const Value>, deque: &mut Vec<ValueCell>) {
    if visited.contains(&(e.as_ptr() as *const Value)) {
        return;
    }
    visited.insert(e.as_ptr() as *const Value);
    for p in e.borrow().prev.iter() {
        build_topo(p, visited, deque);
    }
    deque.push(Rc::clone(e));
}

pub fn backward(vc: &ValueCell) {
    let mut visited = HashSet::new();
    let mut deque = Vec::new();
    build_topo(vc, &mut visited, &mut deque);

    vc.borrow_mut().grad = 1.;
    while let Some(e) = deque.pop() {
        let v = e.borrow();
        if let Some(o) = &v.op {
            match o {
                Op::Add => {
                    v.prev[0].borrow_mut().grad += v.grad;
                    v.prev[1].borrow_mut().grad += v.grad;
                }
                Op::Mul => {
                    v.prev[0].borrow_mut().grad += v.prev[1].borrow().data * v.grad;
                    v.prev[1].borrow_mut().grad += v.prev[0].borrow().data * v.grad;
                }
                Op::Tanh => {
                    v.prev[0].borrow_mut().grad += (1. - v.data.powf(2.)) * v.grad;
                }
                Op::Exp => {
                    v.prev[0].borrow_mut().grad += v.data * v.grad;
                }
                Op::Pow(n) => {
                    // Do not refactor inline, due to mutable borrow
                    let local = n * v.prev[0].borrow().data.powf(n - 1.);
                    v.prev[0].borrow_mut().grad += local * v.grad;
                }
            }
        }
    }
}

// Layers

pub struct NonLinear {
    w: Vec<Vec<ValueCell>>,
    b: Vec<ValueCell>,
}

impl NonLinear {
    pub fn new(inp: usize, out: usize, prefix: &str) -> Result<Self, NormalError> {
        let mut w = Vec::with_capacity(out);
        let mut b = Vec::with_capacity(out);
        for j in 0..out {
            let v = random_vector(
                inp,
                &format!("w{}{}", prefix, j + 1),
                1. / (inp as f32).sqrt(),
            )?;
            w.push(v);
            b.push(lit(0., format!("b{}", j + 1)));
        }
        Ok(NonLinear { w, b })
    }

    pub fn forward(&self, x: &Vec<ValueCell>) -> Vec<ValueCell> {
        let size = self.b.len();
        let mut y = Vec::with_capacity(size);
        for j in 0..size {
            let mut b = Rc::clone(&self.b[j]);
            for (wi, xi) in zip(&self.w[j], x) {
                b = add(&mul(wi, xi), &b);
            }
            let a = tanh(&b);
            y.push(a);
        }
        y
    }

    pub fn parameters(&self) -> Vec<ValueCell> {
        let mut ps = Vec::with_capacity(self.b.len() + self.w.len() * self.w[0].len());
        for bi in self.b.iter() {
            ps.push(Rc::clone(bi));
        }
        for wi in self.w.iter() {
            for wj in wi.iter() {
                ps.push(Rc::clone(wj));
            }
        }
        ps
    }
}

pub struct MLP {
    layers: Vec<NonLinear>,
}

impl MLP {
    pub fn new(sizes: &[usize]) -> Result<Self, NormalError> {
        let mut layers = Vec::with_capacity(sizes.len() - 1);
        for (l, size) in sizes.windows(2).enumerate() {
            let layer = NonLinear::new(size[0], size[1], &format!("{}", l + 1))?;
            layers.push(layer);
        }
        Ok(MLP { layers })
    }

    pub fn forward(&self, xs: &Vec<ValueCell>) -> Vec<ValueCell> {
        let mut ys = self.layers[0].forward(xs);
        for l in 1..self.layers.len() {
            ys = self.layers[l].forward(&ys);
        }
        ys
    }

    pub fn parameters(&self) -> Vec<ValueCell> {
        let mut ps = Vec::new();
        for layer in self.layers.iter() {
            ps.append(&mut layer.parameters());
        }
        ps
    }
}

// Optimizers

pub struct SGD {
    pub params: Vec<ValueCell>,
    pub lr: f32,
}

impl SGD {
    pub fn new(params: Vec<ValueCell>, lr: f32) -> Self {
        SGD { params, lr }
    }

    pub fn zero_grad(&self) {
        for p in self.params.iter() {
            let mut p = p.borrow_mut();
            p.grad = 0.;
        }
    }

    pub fn step(&self) {
        for p in self.params.iter() {
            let g = p.borrow().grad;
            let mut p = p.borrow_mut();
            p.data -= self.lr * g;
        }
    }
}

// SIMD

/// Calculates the dot product between the two 8d vectors `x` and `y`.
pub fn avx2_dot_f32x8(x: &[f32; 8], y: &[f32; 8]) -> f32 {
    let res;
    unsafe {
        let i0 = _mm256_loadu_ps(x.as_ptr());
        let i1 = _mm256_loadu_ps(y.as_ptr());
        let s2 = _mm256_dp_ps::<0xff>(i0, i1);
        let hi = _mm256_extractf128_ps::<1>(s2);
        let lo = _mm256_castps256_ps128(s2);
        let dp = _mm_add_ps(hi, lo);
        res = _mm_extract_ps::<0>(dp);
    }
    f32::from_bits(res as u32)
}

/// Works with 8x8 matrices of floats, multiplying the `a` matrix by the `b` matrix, while accumulating results in `c`.
pub fn fma_matmul_f32x8x8(a: &[f32; 64], b: &[f32; 64], c: &mut [f32; 64]) {
    for i in 0..8 {
        unsafe {
            let b_row = _mm256_loadu_ps(&b[i * 8]);
            let mut c_row = _mm256_loadu_ps(&c[i * 8]);
            for j in 0..8 {
                let a_cel = _mm256_set1_ps(a[i * 8 + j]);
                c_row = _mm256_fmadd_ps(a_cel, b_row, c_row);
            }
            _mm256_storeu_ps(&mut c[i * 8], c_row);
        }
    }
}

// Utilities

fn random_vector(size: usize, prefix: &str, std_dev: f32) -> Result<Vec<ValueCell>, NormalError> {
    let mut rng = thread_rng();
    let normal = Normal::new(0., std_dev)?;
    let mut w = Vec::with_capacity(size);
    for i in 0..size {
        let x = normal.sample(&mut rng);
        w.push(lit(x, format!("{}{}", prefix, i + 1)));
    }
    Ok(w)
}

// Visualization

pub fn graphviz(e: &ValueCell) -> std::io::Result<()> {
    let mut file = File::create("example.dot")?;
    let mut visited = HashSet::new();
    writeln!(file, "digraph {{")?;
    writeln!(file, "  rankdir=\"LR\"")?;
    graphviz_node(e, &mut file, &mut visited)?;
    writeln!(file, "}}")?;
    Ok(())
}

fn graphviz_node(
    e: &ValueCell,
    file: &mut File,
    visited: &mut HashSet<usize>,
) -> std::io::Result<()> {
    let value = e.borrow();
    let p = e.as_ptr() as usize;
    if visited.contains(&p) {
        return Ok(());
    }
    visited.insert(p);
    writeln!(
        file,
        "  \"{}\" [label=\"{} | {{data {:.04} | grad {:.04}}}\", shape=record]",
        p, value.label, value.data, value.grad
    )?;
    if let Some(o) = &value.op {
        graphviz_op(e, file, visited, &o.symbol())?;
    }
    Ok(())
}

fn graphviz_op(
    e: &ValueCell,
    file: &mut File,
    visited: &mut HashSet<usize>,
    symbol: &str,
) -> std::io::Result<()> {
    let value = e.borrow();
    let p = e.as_ptr() as usize;
    writeln!(file, "  \"{:}_op\" [label=\"{}\"]", p, symbol)?;
    writeln!(file, "  \"{:}_op\" -> \"{:}\"", p, p)?;
    for prev in value.prev.iter() {
        graphviz_node(prev, file, visited)?;
        let prev_p = prev.as_ptr() as usize;
        writeln!(file, "  \"{:}\" -> \"{:}_op\"", prev_p, p)?;
    }
    Ok(())
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() -> std::io::Result<()> {
        let a = lit(-2., "a".to_string());
        let b = lit(3., "b".to_string());
        let e = add(&a, &b);
        let d = mul(&a, &b);
        let f = add(&e, &d);
        backward(&f);
        graphviz(&f)?;
        Ok(())
    }

    #[test]
    fn share() -> std::io::Result<()> {
        let a = lit(3., "a".to_string());
        let b = add(&a, &a);
        backward(&b);
        graphviz(&b)?;
        Ok(())
    }

    #[test]
    fn expr() -> std::io::Result<()> {
        let a = lit(2., "a".to_string());
        let b = lit(-3., "b".to_string());
        let c = lit(10., "c".to_string());
        let e = mul(&a, &b);
        let d = add(&e, &c);
        let f = lit(-2., "f".to_string());
        let l = mul(&d, &f);
        let m = tanh(&l);
        backward(&m);
        graphviz(&m)?;
        Ok(())
    }

    #[test]
    fn perceptron() -> std::io::Result<()> {
        let x1 = lit(2., "x1".to_string());
        let x2 = lit(0., "x2".to_string());
        let w1 = lit(-3., "w1".to_string());
        let w2 = lit(1., "w2".to_string());
        let b = lit(6.8813735870195432, "b".to_string());
        let x1w1 = mul(&x1, &w1);
        let x2w2 = mul(&x2, &w2);
        let x1w1x2w2 = add(&x1w1, &x2w2);
        let n = add(&x1w1x2w2, &b);

        let e = exp(&mul(&lit(2., "2".to_string()), &n));
        let o = div(
            &sub(&e, &lit(1., "1".to_string())),
            &add(&e, &lit(1., "1".to_string())),
        );

        backward(&o);
        graphviz(&o)?;

        Ok(())
    }

    #[test]
    fn perceptron_tanh() -> std::io::Result<()> {
        let ref x1 = lit(2., "x1".to_string());
        let ref x2 = lit(0., "x2".to_string());
        let ref w1 = lit(-3., "w1".to_string());
        let ref w2 = lit(1., "w2".to_string());
        let ref b = lit(6.8813735870195432, "b".to_string());
        let ref x1w1 = mul(x1, w1);
        let ref x2w2 = mul(x2, w2);
        let ref x1w1x2w2 = add(x1w1, x2w2);
        let ref n = add(x1w1x2w2, b);

        let ref o = tanh(n);

        backward(o);
        graphviz(o)?;

        Ok(())
    }

    fn nonlinear(
        x: &Vec<ValueCell>,
        size: usize,
        prefix: &str,
    ) -> Result<Vec<ValueCell>, NormalError> {
        let mut y = Vec::with_capacity(size);
        for j in 0..size {
            let w = random_vector(x.len(), &format!("w{}{}", prefix, j + 1), 1.)?;
            let mut b = lit(0., format!("b1{}", j + 1));
            for (wi, xi) in zip(w, x) {
                b = add(&mul(&wi, &xi), &b);
            }
            let a = tanh(&b);
            y.push(a);
        }
        Ok(y)
    }

    #[test]
    fn nn() -> Result<(), NormalError> {
        let (inp, mid, out) = (2, 3, 1);
        let x = random_vector(inp, "x", 1.)?;
        let y = nonlinear(&x, mid, "1")?;
        let z = nonlinear(&y, out, "2")?;
        let o = &z[0];
        backward(&o);
        graphviz(&o).ok();
        Ok(())
    }

    #[test]
    fn layered() -> Result<(), NormalError> {
        let (inp, mid, out) = (2, 3, 1);
        let x = random_vector(inp, "x", 1.)?;
        let y = NonLinear::new(inp, mid, "1")?.forward(&x);
        let z = NonLinear::new(mid, out, "2")?.forward(&y);
        let o = &z[0];
        backward(&o);
        graphviz(&o).ok();
        Ok(())
    }

    #[test]
    fn backprop() -> Result<(), NormalError> {
        let xs = [[2., 3., -1.], [3., -1., 0.5], [0.5, 1., 1.], [1., 1., -1.]];
        let ys = [1., -1., -1., 1.];
        let model = MLP::new(&[3, 4, 4, 1])?;
        let mut loss = lit(0., "0".to_string());
        for (i, (x, y)) in zip(xs, ys).enumerate() {
            let x: Vec<ValueCell> = x
                .iter()
                .enumerate()
                .map(|(j, x)| lit(*x, format!("x{}{}", i + 1, j + 1)))
                .collect();
            let y = lit(y, format!("y{}", i + 1));
            let y_pred = &model.forward(&x)[0];
            let diff = sub(&y, y_pred);
            let local_loss = pow(&diff, 2.);
            loss = add(&loss, &local_loss);
        }
        backward(&loss);
        graphviz(&loss).ok();
        Ok(())
    }

    #[test]
    fn sgd() -> Result<(), NormalError> {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (i, (x, y)) in zip(
            [[2., 3., -1.], [3., -1., 0.5], [0.5, 1., 1.], [1., 1., -1.]],
            [1., -1., -1., 1.],
        )
        .enumerate()
        {
            xs.push(
                x.iter()
                    .enumerate()
                    .map(|(j, x)| lit(*x, format!("x{}{}", i + 1, j + 1)))
                    .collect(),
            );
            ys.push(lit(y, format!("y{}", i + 1)));
        }

        let model = MLP::new(&[3, 4, 4, 1])?;

        let lr = 0.1;
        let opt = SGD::new(model.parameters(), lr);

        let mut loss = lit(0., "0".to_string());
        for _ in 0..20 {
            loss = lit(0., "0".to_string());
            for (x, y) in zip(xs.iter(), ys.iter()) {
                let y_pred = &model.forward(x)[0];
                let diff = sub(y, y_pred);
                let local_loss = pow(&diff, 2.);
                loss = add(&loss, &local_loss);
            }
            println!("Loss = {}", loss.as_ref().borrow().data);

            opt.zero_grad();
            backward(&loss);
            opt.step();
        }

        graphviz(&loss).ok();
        Ok(())
    }

    #[test]
    fn simd_dot_product() {
        if is_x86_feature_detected!("avx2") {
            println!("AVX2 is supported");
        }
        let x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(f32::abs(avx2_dot_f32x8(&x, &y) - 20.400002) < 0.00001);
    }

    #[test]
    fn simd_matmul() {
        if is_x86_feature_detected!("fma") {
            println!("FMA is supported");
        }
        let a = [2.0; 64];
        let mut c = [0.0; 64];
        fma_matmul_f32x8x8(&a, &a, &mut c);
        for i in 0..8 {
            for j in 0..8 {
                if j > 0 {
                    print!(" ");
                }
                let cel = &c[i * 8 + j];
                print!("{}", cel);
                assert!(f32::abs(cel - 32.0) < 0.00001);
            }
            println!();
        }
    }
}
