use std::sync::Arc;
use std::ops::{Add, AddAssign};
use std::fmt;

const MIN_CHILDREN: usize = 2;
const MAX_CHILDREN: usize = 4;

pub trait Item: Clone + Eq + fmt::Debug {
    type Summary: for<'a> AddAssign<&'a Self::Summary> + Default + Clone + Eq + fmt::Debug;

    fn summarize(&self) -> Self::Summary;
}

pub trait Dimension: for<'a> Add<&'a Self, Output = Self> + Clone + Ord + fmt::Debug {
    type Summary: Default + Clone + Eq + fmt::Debug;

    fn from_summary(summary: &Self::Summary) -> Self;

    fn default() -> Self {
        Self::from_summary(&Self::Summary::default())
    }
}

/// Thread safe tree data structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tree<I: Item>(Arc<Node<I>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node<I: Item> {
    Branch {
        rightmost_leaf: Option<Tree<I>>,
        summary: I::Summary,
        children: Vec<Tree<I>>,
        height: u16,
    },
    Leaf {
        summary: I::Summary,
        value: I,
    }
}

pub struct Iter<'a, I: Item + 'a> {
    tree: &'a Tree<I>,
    did_start: bool,
    stack: Vec<(&'a Tree<I>, usize)>,
}

pub struct Cursor<'a, I: Item + 'a> {
    tree: &'a Tree<I>,
    did_seek: bool,
    stack: Vec<(&'a Tree<I>, usize, I::Summary)>,
    prev_leaf: Option<&'a Tree<I>>,
    summary: I::Summary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekBias {
    Left,
    Right,
}

impl<'a, I: Item> Tree<I> {
    pub fn new() -> Tree<I> {
        Tree::from_children(Vec::new())
    }

    pub fn from_item(item: I) -> Tree<I> {
        let mut tree = Tree::new();

        tree.push(item);
        tree
    }

    fn from_children(children: Vec<Tree<I>>) -> Tree<I> {
        let summary = Self::summarize_children(&children);
        let rightmost_leaf = children
            .last()
            .and_then(|last_child| last_child.rightmost_leaf().cloned());
        let height = children.get(0).map(|c| c.height()).unwrap_or(0) + 1;

        Tree(Arc::new(Node::Branch {
            rightmost_leaf,
            summary,
            children,
            height,
        }))
    }

    fn summarize_children(children: &[Tree<I>]) -> I::Summary {
        let mut summary = I::Summary::default();

        for child in children.iter() {
            summary += child.summary();
        }

        summary
    }

    pub fn iter(&self) -> Iter<I> {
        Iter::new(self)
    }

    pub fn cursor(&self) -> Cursor<I> {
        Cursor::new(self)
    }

    pub fn len<D: Dimension<Summary = I::Summary>>(&self) -> D {
        D::from_summary(self.summary())
    }

    pub fn last(&self) -> Option<&I> {
        self.rightmost_leaf().map(|leaf| leaf.value())
    }

    pub fn push(&mut self, item: I) {
        self.push_tree(Tree(Arc::new(Node::Leaf {
            summary: item.summarize(),
            value: item,
        })));
    }

    pub fn push_tree(&mut self, other: Tree<I>) {
        if other.is_empty() {
            return;
        }

        let self_height = self.height();
        let other_height = other.height();

        if self_height < other_height {
            for other_child in other.children().iter().cloned() {
                self.push_tree(other_child);
            }

            return;
        }

        if let Some(split) = self.push_recursive(other) {
            *self = Tree::from_children(vec![self.clone(), split]);
        }
    }

    fn push_recursive(&mut self, other: Tree<I>) -> Option<Tree<I>> {
        *self.summary_mut() += other.summary();
        *self.rightmost_leaf_mut() = other.rightmost_leaf().cloned();

        let self_height = self.height();
        let other_height = other.height();

        if other_height == self_height {
            self.append_children(other.children())
        } else if other_height ==  self_height - 1 && !other.underflowing() {
            self.append_children(&[other])
        } else {
            if let Some(split) = self.last_child_mut().push_recursive(other) {
                self.append_children(&[split])
            } else {
                None
            }
        }
    }

    fn append_children(&mut self, new_children: &[Tree<I>]) -> Option<Tree<I>> {
        match Arc::make_mut(&mut self.0) {
            Node::Branch { children, summary, rightmost_leaf, .. } => {
                let child_count = children.len() + new_children.len();

                if child_count > MAX_CHILDREN {
                    let midpoint = (child_count + child_count % 2) / 2;
                    let (left_children, right_children) = {
                        let mut all_children = children.iter().chain(new_children.iter()).cloned();

                        (
                            all_children.by_ref().take(midpoint).collect(),
                            all_children.collect()
                        )
                    };

                    *children = left_children;
                    *summary = Tree::summarize_children(children);
                    *rightmost_leaf = children.last().unwrap().rightmost_leaf().cloned();

                    Some(Tree::from_children(right_children))
                } else {
                    children.extend(new_children.iter().cloned());

                    None
                }
            },
            Node::Leaf { .. } => unreachable!()
        }
    }

    fn rightmost_leaf(&self) -> Option<&Tree<I>> {
        match self.0.as_ref() {
            Node::Branch { rightmost_leaf, .. } => rightmost_leaf.as_ref(),
            Node::Leaf { .. } => Some(self),
        }
    }

    fn rightmost_leaf_mut(&mut self) -> &mut Option<Tree<I>> {
        match Arc::make_mut(&mut self.0) {
            Node::Branch { rightmost_leaf, .. } => rightmost_leaf,
            Node::Leaf { .. } => unreachable!(),
        }
    }

    pub fn summary(&self) -> &I::Summary {
        match self.0.as_ref() {
            Node::Branch { summary, .. } => summary,
            Node::Leaf { summary, .. } => summary,
        }
    }

    fn summary_mut(&mut self) -> &mut I::Summary {
        match Arc::make_mut(&mut self.0) {
            Node::Branch { summary, .. } => summary,
            Node::Leaf { summary, .. } => summary,
        }
    }

    fn children(&self) -> &[Tree<I>] {
        match self.0.as_ref() {
            Node::Branch { children, .. } => children.as_slice(),
            Node::Leaf { .. } => unreachable!(),
        }
    }

    fn last_child_mut(&mut self) -> &mut Tree<I> {
        match Arc::make_mut(&mut self.0) {
            Node::Branch { children, .. } => children.last_mut().unwrap(),
            Node::Leaf { .. } => unreachable!(),
        }
    }

    fn value(&self) -> &I {
        match self.0.as_ref() {
            Node::Branch { .. } => unreachable!(),
            Node::Leaf { value, .. } => value,
        }
    }

    fn underflowing(&self) -> bool {
        match self.0.as_ref() {
            Node::Branch { children, .. } => children.len() < MIN_CHILDREN,
            Node::Leaf { .. } => false,
        }
    }

    fn is_empty(&self) -> bool {
        match self.0.as_ref() {
            Node::Branch { children, .. } => children.is_empty(),
            Node::Leaf { .. } => false,
        }
    }

    fn height(&self) -> u16 {
        match self.0.as_ref() {
            &Node::Branch { height, .. } => height,
            Node::Leaf { .. } => 0,
        }
    }
}

impl<I: Item> Extend<I> for Tree<I> {
    fn extend<II: IntoIterator<Item = I>>(&mut self, items: II) {
        for item in items.into_iter() {
            self.push(item);
        }
    }
}

impl<'a, I: Item + 'a> Iter<'a, I> {
    fn new(tree: &'a Tree<I>) -> Iter<'a, I> {
        Iter {
            tree,
            did_start: false,
            stack: Vec::with_capacity(tree.height() as usize),
        }
    }

    fn seek_to_first_item(&mut self, mut tree: &'a Tree<I>) -> Option<&'a I> {
        if tree.is_empty() {
            None
        } else {
            loop {
                match tree.0.as_ref() {
                    Node::Branch { children, .. } => {
                        self.stack.push((tree, 0));
                        tree = &children[0];
                    },
                    Node::Leaf { value, .. } => return Some(value),
                }
            }
        }
    }
}

impl<'a, I: Item + 'a> Iterator for Iter<'a, I> {
    type Item = &'a I;

    fn next(&mut self) -> Option<&'a I> {
        if self.did_start {
            while self.stack.len() > 0 {
                let (tree, index) = {
                    let (tree, index) = self.stack.last_mut().unwrap();

                    *index += 1;

                    (tree, *index)
                };

                if let Some(child) = tree.children().get(index) {
                    return self.seek_to_first_item(child);
                } else {
                    self.stack.pop();
                }
            }

            None
        } else {
            self.did_start = true;
            self.seek_to_first_item(self.tree)
        }
    }
}

impl<'a, I: Item + 'a> Cursor<'a, I> {
    fn new(tree: &'a Tree<I>) -> Cursor<'a, I> {
        Cursor {
            tree,
            did_seek: false,
            stack: Vec::with_capacity(tree.height() as usize),
            prev_leaf: None,
            summary: I::Summary::default(),
        }
    }

    fn reset(&mut self) {
        self.did_seek = false;
        self.stack.truncate(0);
        self.prev_leaf = None;
        self.summary = I::Summary::default();
    }

    pub fn start<D: Dimension<Summary = I::Summary>>(&self) -> D {
        D::from_summary(&self.summary)
    }

    pub fn item<'b>(&'b self) -> Option<&'a I> {
        self.cur_leaf().map(|leaf| leaf.value())
    }

    pub fn prev_item<'b>(&'b self) -> Option<&'a I> {
        self.prev_leaf.map(|leaf| leaf.value())
    }

    pub fn cur_leaf<'b>(&'b self) -> Option<&'a Tree<I>> {
        self.stack.last().map(|&(subtree, index, _)| &subtree.children()[index])
    }

    pub fn next(&mut self) {
        while self.stack.len() > 0 {
            let (prev_subtree, index) = {
                let (prev_subtree, index, _) = self.stack.last_mut().unwrap();

                if prev_subtree.height() == 1 {
                    let prev_leaf = &prev_subtree.children()[*index];

                    self.prev_leaf = Some(prev_leaf);
                    self.summary += prev_leaf.summary();
                }

                *index += 1;

                (prev_subtree, *index)
            };

            if let Some(child) = prev_subtree.children().get(index) {
                self.seek_to_first_item(child);

                break;
            } else {
                self.stack.pop();
            }
        }
    }

    pub fn prev(&mut self) {
        if self.stack.is_empty() && self.prev_leaf.is_some() {
            self.summary = I::Summary::default();
            self.seek_to_last_item(self.tree);
        } else {
            while self.stack.len() > 0 {
                let subtree = {
                    let (parent, index, summary) = self.stack.last_mut().unwrap();

                    if *index == 0 {
                        None
                    } else {
                        *index -= 1;
                        self.summary = summary.clone();

                        for child in &parent.children()[0..*index] {
                            self.summary += child.summary();
                        }

                        parent.children().get(*index)
                    }
                };

                if let Some(subtree) = subtree {
                    self.seek_to_last_item(subtree);

                    break;
                } else {
                    self.stack.pop();
                }
            }
        }

        self.prev_leaf = if self.stack.is_empty() {
            None
        } else {
            let mut stack_index = self.stack.len() - 1;

            loop {
                let (ancestor, index, _) = &self.stack[stack_index];

                if *index == 0 {
                    if stack_index == 0 {
                        break None;
                    } else {
                        stack_index -= 1;
                    }
                } else {
                    break ancestor.children()[index - 1].rightmost_leaf();
                }
            }
        };
    }

    fn seek_to_first_item<'b>(&'b mut self, mut tree: &'a Tree<I>) {
        self.did_seek = true;

        loop {
            match tree.0.as_ref() {
                Node::Branch { children, .. } => {
                    self.stack.push((tree, 0, self.summary.clone()));
                    tree = &children[0];
                },
                Node::Leaf { .. } => break,
            }
        }
    }

    fn seek_to_last_item<'b>(&'b mut self, mut tree: &'a Tree<I>) {
        self.did_seek = true;

        loop {
            match tree.0.as_ref() {
                Node::Branch { children, .. } => {
                    self.stack.push((tree, children.len() - 1, self.summary.clone()));

                    for child in &tree.children()[0..children.len() - 1] {
                        self.summary += child.summary();
                    }

                    tree = children.last().unwrap();
                },
                Node::Leaf { .. } => break,
            }
        }
    }

    pub fn seek<D: Dimension<Summary = I::Summary>>(&mut self, pos: &D, bias: SeekBias) {
        self.reset();
        self.seek_and_slice(pos, bias, None);
    }

    pub fn slice<D: Dimension<Summary = I::Summary>>(&mut self, end: &D, bias: SeekBias) -> Tree<I> {
        let mut prefix = Tree::new();

        self.seek_and_slice(end, bias, Some(&mut prefix));

        prefix
    }

    fn seek_and_slice<D: Dimension<Summary = I::Summary>>(&mut self, pos: &D, bias: SeekBias, mut slice: Option<&mut Tree<I>>) {
        let mut cur_subtree = None;

        if self.did_seek {
            while self.stack.len() > 0 {
                {
                    let (prev_subtree, index, _) = self.stack.last_mut().unwrap();

                    if prev_subtree.height() > 1 {
                        *index += 1;
                    }

                    let children_len = prev_subtree.children().len();

                    while *index < children_len {
                        let subtree = &prev_subtree.children()[*index];
                        let summary = subtree.summary();
                        let subtree_end = D::from_summary(&self.summary) + &D::from_summary(summary);

                        if *pos > subtree_end || (*pos == subtree_end && bias == SeekBias::Right) {
                            self.summary += summary;
                            self.prev_leaf = subtree.rightmost_leaf();
                            slice.as_mut().map(|slice| slice.push_tree(subtree.clone()));
                            *index += 1;
                        } else {
                            cur_subtree = Some(subtree);

                            break;
                        }
                    }
                }

                if cur_subtree.is_some() {
                    break;
                } else {
                    self.stack.pop();
                }
            }
        } else {
            self.reset();
            self.did_seek = true;
            cur_subtree = Some(self.tree);
        }

        while let Some(subtree) = cur_subtree.take() {
            match subtree.0.as_ref() {
                Node::Branch { rightmost_leaf, summary, children, .. } => {
                    let subtree_end = D::from_summary(&self.summary) + &D::from_summary(summary);

                    if *pos > subtree_end || (*pos == subtree_end && bias == SeekBias::Right) {
                        self.summary += summary;
                        self.prev_leaf = rightmost_leaf.as_ref();
                        slice.as_mut().map(|slice| slice.push_tree(subtree.clone()));
                    } else {
                        for (index, child) in children.iter().enumerate() {
                            let child_end = D::from_summary(&self.summary) + &D::from_summary(child.summary());

                            if *pos > child_end || (*pos == child_end && bias == SeekBias::Right) {
                                self.summary += child.summary();
                                self.prev_leaf = child.rightmost_leaf();
                                slice.as_mut().map(|slice| slice.push_tree(child.clone()));
                            } else {
                                self.stack.push((subtree, index, self.summary.clone()));
                                cur_subtree = Some(child);

                                break;
                            }
                        }
                    }
                },
                Node::Leaf { summary, .. } => {
                    let subtree_end = D::from_summary(&self.summary) + &D::from_summary(summary);

                    if *pos > subtree_end || (*pos == subtree_end && bias == SeekBias::Right) {
                        self.prev_leaf = Some(subtree);
                        self.summary += summary;
                        slice.as_mut().map(|slice| slice.push_tree(subtree.clone()));
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    pub struct IntSummary {
        count: usize,
        sum: usize,
    }

    #[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct Count(usize);

    #[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct Sum(usize);

    impl Item for usize {
        type Summary = IntSummary;

        fn summarize(&self) -> IntSummary {
            IntSummary {
                count: 1,
                sum: *self
            }
        }
    }

    impl<'a> AddAssign<&'a IntSummary> for IntSummary {
        fn add_assign(&mut self, other: &IntSummary) {
            self.count += other.count;
            self.sum += other.sum;
        }
    }

    impl Dimension for Count {
        type Summary = IntSummary;

        fn from_summary(summary: &IntSummary) -> Count {
            Count(summary.count)
        }
    }

    impl Dimension for Sum {
        type Summary = IntSummary;

        fn from_summary(summary: &IntSummary) -> Sum {
            Sum(summary.sum)
        }
    }

    impl<'a> Add<&'a Count> for Count {
        type Output = Count;

        fn add(mut self, other: &Count) -> Count {
            self.0 += other.0;
            self
        }
    }

    impl<'a> Add<&'a Sum> for Sum {
        type Output = Sum;

        fn add(mut self, other: &Sum) -> Sum {
            self.0 += other.0;
            self
        }
    }

    impl<I: Item> Tree<I> {
        fn items(&self) -> Vec<I> {
            self.iter().cloned().collect()
        }
    }

    #[test]
    fn test_extend_and_push() {
        use std::thread;

        let mut tree1 = Tree::new();
        let mut tree2 = Tree::new();

        let mut tree1 = thread::spawn(move || {
            tree1.extend(1..=20);
            tree1
        }).join().unwrap();

        let tree2 = thread::spawn(move || {
            tree2.extend(1..=50);
            tree2
        }).join().unwrap();

        tree1.push_tree(tree2);

        assert_eq!(tree1.items(), (1..=20).chain(1..=50).collect::<Vec<usize>>());
        assert_eq!(tree1.len::<Count>(), Count(70));
        assert_eq!(tree1.len::<Sum>(), Sum(1485));
    }
}
