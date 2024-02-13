use std::{fmt, rc::Rc};

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

fn contiguous_stride(size: &[usize]) -> Vec<isize> {
    let mut stride: Vec<isize> = size
        .iter()
        .rev()
        .scan(1, |dim_prod, &dim_size| {
            let new_dim_size = *dim_prod;
            *dim_prod *= dim_size as isize;
            Some(new_dim_size)
        })
        .collect();
    stride.reverse();
    stride
}

impl StridedTensor {
    fn new(size: &[usize], data: &[f32]) -> Self {
        let numel = size.iter().product::<usize>();
        assert_eq!(
            numel,
            data.len(),
            "Expected data with {} elements when creating tensor of size {:?}, but got {}",
            numel,
            size,
            data.len()
        );
        Self {
            storage: Rc::new(data.to_vec()),
            storage_offset: 0,
            stride: contiguous_stride(size),
            size: size.to_vec(),
        }
    }

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

    fn elem_offset(&self, indices: &[usize]) -> usize {
        (self.storage_offset as isize
            + indices
                .iter()
                .zip(self.stride.iter())
                .map(|(&i, &s)| i as isize * s)
                .sum::<isize>()) as usize
    }

    fn unchecked_elem(&self, indices: &[usize]) -> f32 {
        self.storage[self.elem_offset(indices)]
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
        let index = self.elem_offset(indices);
        assert!(
            index < self.storage.len(),
            "Index {} out of range for storage of size {}",
            index,
            self.storage.len()
        );
        self.storage[index]
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

    fn check_dim(&self, dim: usize) {
        let rank = self.size.len();
        assert!(
            dim < rank,
            "Dimension out of range (expected to be in range of [0, {}], but got {})",
            rank - 1,
            dim
        );
    }

    fn transpose_(&mut self, dim0: usize, dim1: usize) {
        self.check_dim(dim0);
        self.check_dim(dim1);
        if dim0 == dim1 {
            return;
        }
        self.stride.swap(dim0, dim1);
        self.size.swap(dim0, dim1);
    }

    fn t(&self) -> Self {
        let rank = self.size.len();
        assert_eq!(
            rank, 2,
            "Matrix transposition expects 2 dimensions, but got {}",
            rank
        );
        self.transpose(0, 1)
    }

    fn t_(&mut self) {
        let rank = self.size.len();
        assert_eq!(
            rank, 2,
            "Matrix transposition expects 2 dimensions, but got {}",
            rank
        );
        self.transpose_(0, 1);
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
        let new_stride: Vec<isize> = contiguous_stride(&self.size);
        let mut new_storage = Vec::with_capacity(self.numel());
        self.extend_vec(&mut new_storage);
        Self {
            storage: Rc::new(new_storage),
            storage_offset: 0,
            stride: new_stride,
            size: self.size.clone(),
        }
    }

    fn flip(&self, dim: usize) -> Self {
        self.check_dim(dim);
        let mut new_stride = self.stride.clone();
        new_stride[dim] *= -1;
        let offset =
            self.storage_offset as isize + self.stride[dim] * (self.size[dim] - 1) as isize;
        Self {
            storage: Rc::clone(&self.storage),
            storage_offset: offset as usize,
            stride: new_stride,
            size: self.size.clone(),
        }
    }

    fn full(size: &[usize], value: f32) -> Self {
        Self {
            storage: Rc::new(vec![value]),
            storage_offset: 0,
            stride: vec![0; size.len()],
            size: size.to_vec(),
        }
    }

    fn zeros(size: &[usize]) -> Self {
        Self::full(size, Default::default())
    }

    fn ones(size: &[usize]) -> Self {
        Self::full(size, 1.0)
    }

    fn dot(&self, other: &Self) -> f32 {
        assert!(
            self.size.len() == 1 && other.size.len() == 1,
            "1D tensors expected, but got {}D and {}D tensors",
            self.size.len(),
            other.size.len()
        );
        assert!(
            self.size[0] == other.size[0],
            "Inconsistent tensor size, expected tensor [{}] and src [{}] to have the same number of elements, but got {} and {} elements respectively",
            self.size[0], other.size[0], self.size[0], other.size[0]
        );
        self.iter().zip(other.iter()).map(|(x, y)| x * y).sum()
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
                Self {
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

struct StridedTensorIterator<'a> {
    tensor: &'a StridedTensor,
    index: Option<Vec<usize>>,
}

impl<'a> StridedTensorIterator<'a> {
    fn new(tensor: &'a StridedTensor) -> Self {
        Self {
            tensor: tensor,
            index: Some(vec![0; tensor.size.len()]),
        }
    }
}

impl<'a> Iterator for StridedTensorIterator<'a> {
    type Item = f32;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = &mut self.index {
            let elem = self.tensor.elem(index);
            if let Some((i, _)) = index
                .iter()
                .zip(self.tensor.size.iter())
                .enumerate()
                .rev()
                .find(|(_, (&d, &s))| d < s - 1)
            {
                index[i] += 1;
                for j in i + 1..index.len() {
                    index[j] = 0;
                }
            } else {
                self.index = None
            }
            return Some(elem);
        }
        None
    }
}

impl StridedTensor {
    fn iter(&self) -> StridedTensorIterator {
        StridedTensorIterator::new(&self)
    }
}

impl<'a> IntoIterator for &'a StridedTensor {
    type Item = f32;
    type IntoIter = StridedTensorIterator<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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
    #[should_panic]
    fn non_matrix_transpose() {
        tensor_example_3().t();
    }

    #[test]
    fn matrix_transpose() {
        assert_eq!(tensor_example_1().t().to_string(), "[[1, 3], [2, 4]]");
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

    #[test]
    fn elem_iterator() {
        let x = tensor_example_1();
        let mut it = x.iter();
        assert_eq!(it.next(), Some(1.0));
        assert_eq!(it.next(), Some(2.0));
        assert_eq!(it.next(), Some(3.0));
        assert_eq!(it.next(), Some(4.0));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn tensor_into_iter() {
        let x = tensor_example_1();
        let mut elems = 0;
        for elem in &x {
            assert!(elem > 0.0 && elem < 5.0);
            elems += 1;
        }
        assert_eq!(elems, x.numel());
    }

    #[test]
    fn tensor_flip() {
        let storage = Rc::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let x = StridedTensor {
            storage: Rc::clone(&storage),
            storage_offset: 1,
            stride: vec![3, 1],
            size: vec![3, 3],
        };
        assert_eq!(x.flip(0).to_string(), "[[7, 8, 9], [4, 5, 6], [1, 2, 3]]");
        assert_eq!(x.flip(1).to_string(), "[[3, 2, 1], [6, 5, 4], [9, 8, 7]]");
        assert_eq!(
            x.flip(0).flip(1).to_string(),
            "[[9, 8, 7], [6, 5, 4], [3, 2, 1]]"
        );
        assert_eq!(x.to_string(), x.flip(0).flip(0).to_string());
        assert_eq!(x.to_string(), x.flip(1).flip(1).to_string());
        assert_eq!(x.to_string(), x.flip(0).flip(1).flip(0).flip(1).to_string());
        let y = StridedTensor {
            storage: Rc::clone(&storage),
            storage_offset: 1,
            stride: vec![4, 2],
            size: vec![2, 2],
        };
        assert_eq!(y.flip(0).to_string(), "[[5, 7], [1, 3]]");
        assert_eq!(y.flip(1).to_string(), "[[3, 1], [7, 5]]");
        assert_eq!(y.flip(0).flip(1).to_string(), "[[7, 5], [3, 1]]");
        let z = StridedTensor {
            storage: Rc::clone(&storage),
            storage_offset: 1,
            stride: vec![4, 2, 1],
            size: vec![2, 2, 2],
        };
        assert_eq!(
            z.flip(0).to_string(),
            "[[[5, 6], [7, 8]], [[1, 2], [3, 4]]]"
        );
        assert_eq!(
            z.flip(1).to_string(),
            "[[[3, 4], [1, 2]], [[7, 8], [5, 6]]]"
        );
        assert_eq!(
            z.flip(2).to_string(),
            "[[[2, 1], [4, 3]], [[6, 5], [8, 7]]]"
        );
    }

    #[test]
    #[should_panic]
    fn new_tensor_with_missing_data() {
        StridedTensor::new(&[2, 2], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn new_tensor() {
        assert_eq!(
            StridedTensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).to_string(),
            "[[1, 2], [3, 4]]"
        );
    }

    #[test]
    fn full_tensor() {
        assert_eq!(StridedTensor::zeros(&[3]).to_string(), "[0, 0, 0]");
        assert_eq!(StridedTensor::ones(&[2]).to_string(), "[1, 1]");
        assert_eq!(
            StridedTensor::full(&[2, 2], 2.0).to_string(),
            "[[2, 2], [2, 2]]"
        );
    }

    #[test]
    #[should_panic]
    fn dot_product_with_non_vectors() {
        tensor_example_1().dot(&tensor_example_3());
    }

    #[test]
    #[should_panic]
    fn dot_product_with_different_sizes() {
        let x = StridedTensor {
            storage: Rc::new(vec![1.0, 2.0, 3.0]),
            storage_offset: 0,
            stride: vec![1],
            size: vec![3],
        };
        let y = StridedTensor {
            storage: Rc::new(vec![4.0, 5.0]),
            storage_offset: 0,
            stride: vec![1],
            size: vec![2],
        };
        x.dot(&y);
    }

    #[test]
    fn dot_product() {
        let x = StridedTensor {
            storage: Rc::new(vec![1.0, 2.0, 3.0]),
            storage_offset: 0,
            stride: vec![1],
            size: vec![3],
        };
        assert_eq!(x.dot(&x), 14.0);
    }
}
