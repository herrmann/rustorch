use std::cell::RefCell;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::rc::Rc;

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
    data: f32,
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

// Main

fn main() {}

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
}
