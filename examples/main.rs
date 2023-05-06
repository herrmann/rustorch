use rustorch::{add, backward, graphviz, lit, pow, sub, MLP, SGD};
use std::iter::zip;

fn main() {
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

    let model = MLP::new(&[3, 4, 4, 1]).unwrap();

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
}
