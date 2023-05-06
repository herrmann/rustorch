use std::{rc::Rc, fmt};

#[derive(Debug)]
struct StridedTensor {
    storage: Rc<Vec<f32>>,
    storage_offset: usize,
    stride: Vec<isize>,
    size: Vec<usize>,
}

impl Default for StridedTensor {
    fn default() -> Self {
        Self {
            storage: Rc::new(vec![f32::default()]),
            storage_offset: 0,
            stride: vec![],
            size: vec![],
        }
    }
}

impl StridedTensor {
    fn element_size(&self) -> usize {
        std::mem::size_of::<f32>()
    }

    fn numel(&self) -> usize {
        self.size.iter().product()
    }

    fn item(&self) -> f32 {
        let numel = self.numel();
        assert_eq!(
            numel, 1,
            "A Tensor with {} elements cannot be converted to Scalar",
            numel
        );
        self.storage[self.storage_offset]
    }

    fn elem(&self, indices: &[usize]) -> f32 {
        assert!(
            !self.size.is_empty(),
            "Invalid index of a 0-dim tensor. Use `tensor.item()`"
        );
        assert_eq!(
            indices.len(),
            self.size.len(),
            "Indices differ for tensor of dimension {}",
            self.size.len()
        );
        for (d, (i, s)) in indices.iter().zip(self.size.iter()).enumerate() {
            #[rustfmt::skip]
            assert!(i < s, "Index {} is out of bounds for dimension {} with size {}", i, d, s);
        }
        let index = self.storage_offset as isize
            + indices
                .iter()
                .zip(self.stride.iter())
                .map(|(&i, &s)| i as isize * s)
                .sum::<isize>();
        assert!(
            index >= 0 && (index as usize) < self.storage.len(),
            "Index {} out of range for storage of size {}",
            index,
            self.storage.len()
        );
        self.storage[index as usize]
    }

    fn index(&self, indices: &[usize]) -> Self {
        assert!(
            indices.len() <= self.size.len(),
            "Too many indices for tensor of dimension {}",
            self.size.len()
        );
        todo!();
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let rank = self.size.len();
        let check_dim = |rank, dim| {
            assert!(
                dim < rank,
                "Dimension out of range (expected to be in range of [0, {}], but got {})",
                rank - 1,
                dim
            );
        };
        check_dim(rank, dim0);
        check_dim(rank, dim1);
        if dim0 == dim1 {
            return Self {
                storage: Rc::clone(&self.storage),
                storage_offset: self.storage_offset,
                stride: self.stride.clone(),
                size: self.size.clone(),
            };
        }
        let mut stride_t = self.stride.clone();
        let mut size_t = self.size.clone();
        stride_t.swap(dim0, dim1);
        size_t.swap(dim0, dim1);
        Self {
            storage: Rc::clone(&self.storage),
            storage_offset: self.storage_offset,
            stride: stride_t,
            size: size_t,
        }
    }

    fn transpose_(&mut self, dim0: usize, dim1: usize) {
        let rank = self.size.len();
        let check_dim = |rank, dim| {
            assert!(
                dim < rank,
                "Dimension out of range (expected to be in range of [0, {}], but got {})",
                rank - 1,
                dim
            );
        };
        check_dim(rank, dim0);
        check_dim(rank, dim1);
        if dim0 == dim1 {
            return;
        }
        self.stride.swap(dim0, dim1);
        self.size.swap(dim0, dim1);
    }

    fn is_contiguous(&self) -> bool {
        if self.numel() < 2 {
            return true;
        }
        let mut expected_stride = 1;
        for (x, y) in self.size.iter().zip(self.stride.iter()).rev() {
            if *x == 1 {
                continue;
            }
            if *y != expected_stride {
                return false;
            }
            expected_stride *= *x as isize;
        }
        true
    }

    fn extend_vec(&self, storage: &mut Vec<f32>) {
        if self.size.len() == 1 {
            for i in 0..self.size[0] {
                let offset = self.storage_offset as isize + i as isize * self.stride[0];
                storage.push(self.storage[offset as usize]);
            }
        } else {
            for i in 0..self.size[0] {
                let offset = self.storage_offset as isize + i as isize * self.stride[0];
                Self {
                    storage: Rc::clone(&self.storage),
                    storage_offset: offset as usize,
                    stride: self.stride[1..].to_vec(),
                    size: self.size[1..].to_vec(),
                }
                .extend_vec(storage);
            }
        }
    }

    fn contiguous(&self) -> Self {
        let mut new_stride: Vec<isize> = self
            .size
            .iter()
            .rev()
            .scan(1, |dim_prod, &dim_size| {
                let new_dim_size = *dim_prod;
                *dim_prod *= dim_size as isize;
                Some(new_dim_size)
            })
            .collect();
        new_stride.reverse();
        let mut new_storage = Vec::with_capacity(self.numel());
        self.extend_vec(&mut new_storage);
        Self {
            storage: Rc::new(new_storage),
            storage_offset: 0,
            stride: new_stride,
            size: self.size.clone(),
        }
    }
}

impl fmt::Display for StridedTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.size.is_empty() {
            write!(f, "{}", self.item())
        } else {
            write!(f, "[")?;
            for i in 0..self.size[0] {
                if i > 0 {
                    write!(f, ", ")?;
                }
                #[rustfmt::skip]
                StridedTensor {
                    storage: Rc::clone(&self.storage),
                    storage_offset: (self.storage_offset as isize + i as isize * self.stride[0]) as usize,
                    stride: self.stride[1..].to_vec(),
                    size: self.size[1..].to_vec(),
                }
                .fmt(f)?
            }
            write!(f, "]")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_scalar(x: f32) -> StridedTensor {
        StridedTensor {
            storage: Rc::new(vec![x]),
            ..Default::default()
        }
    }

    fn tensor_example_1() -> StridedTensor {
        let storage = Rc::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        StridedTensor {
            storage: storage,
            storage_offset: 1,
            stride: vec![2, 1],
            size: vec![2, 2],
        }
    }

    fn tensor_example_2() -> StridedTensor {
        let storage = Rc::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        StridedTensor {
            storage: storage,
            storage_offset: 0,
            stride: vec![3, 2],
            size: vec![2, 2],
        }
    }

    fn tensor_example_3() -> StridedTensor {
        let storage = Rc::new(vec![
            1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0,
        ]);
        StridedTensor {
            storage: storage,
            storage_offset: 0,
            stride: vec![6, 3, 1],
            size: vec![2, 2, 3],
        }
    }

    #[test]
    fn tensor_item() {
        assert_eq!(tensor_scalar(42.0).item(), 42.0);
    }

    #[test]
    fn tensor_display() {
        assert_eq!(tensor_example_1().to_string(), "[[1, 2], [3, 4]]");
        assert_eq!(tensor_example_2().to_string(), "[[0, 2], [3, 5]]");
    }

    #[test]
    #[should_panic]
    fn tensor_not_scalar() {
        tensor_example_1().item();
    }

    #[test]
    #[should_panic]
    fn tensor_index_many_indices() {
        tensor_example_1().index(&[0, 1, 2]);
    }

    #[test]
    #[should_panic]
    fn tensor_index_out_of_bounds() {
        tensor_example_1().index(&[0, 2]);
    }

    #[test]
    fn tensor_stride() {
        assert_eq!(tensor_example_1().elem(&[1, 1]), 4.0);
    }

    #[test]
    #[should_panic]
    fn transposition_wrong_rank_1() {
        tensor_example_3().transpose(0, 3);
    }

    #[test]
    #[should_panic]
    fn transposition_wrong_rank_2() {
        tensor_example_3().transpose(3, 0);
    }

    #[test]
    fn transposition() {
        let x = tensor_example_3();
        assert_eq!(
            x.to_string(),
            "[[[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]]]"
        );
        assert_eq!(
            x.transpose(0, 1).to_string(),
            "[[[1, 2, 3], [5, 6, 7]], [[3, 4, 5], [7, 8, 9]]]"
        );
        assert_eq!(
            x.transpose(0, 2).to_string(),
            "[[[1, 5], [3, 7]], [[2, 6], [4, 8]], [[3, 7], [5, 9]]]"
        );
        assert_eq!(
            x.transpose(1, 2).to_string(),
            "[[[1, 3], [2, 4], [3, 5]], [[5, 7], [6, 8], [7, 9]]]"
        );
        assert_eq!(x.transpose(0, 1).to_string(), x.transpose(1, 0).to_string());
        assert_eq!(x.transpose(0, 2).to_string(), x.transpose(2, 0).to_string());
        assert_eq!(x.transpose(1, 2).to_string(), x.transpose(2, 1).to_string());
    }

    #[test]
    fn inplace_transposition() {
        let mut x = tensor_example_3();
        x.transpose_(0, 1);
        assert_eq!(
            x.to_string(),
            "[[[1, 2, 3], [5, 6, 7]], [[3, 4, 5], [7, 8, 9]]]"
        );
    }

    #[test]
    fn contiguous_testing() {
        assert!(tensor_example_1().is_contiguous());
        assert!(!tensor_example_2().is_contiguous());
        assert!(tensor_example_3().is_contiguous());
        assert!(!tensor_example_3().transpose(1, 2).is_contiguous());
    }

    #[test]
    fn make_contiguous() {
        let x = tensor_example_3();
        assert!(x.is_contiguous());
        let y = x.transpose(0, 1);
        assert!(!y.is_contiguous());
        let z = y.contiguous();
        assert!(z.is_contiguous());
        assert_eq!(z.numel(), z.storage.len());
    }
}
