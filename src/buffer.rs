use crate::tree::{self, Tree};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::ops::{Add, AddAssign, Range};

pub type ReplicaId = usize;
pub type BufferId = usize;
type LocalTimestamp = usize;
type LamportTimeStamp = usize;

pub struct Buffer {
    pub replica_id: ReplicaId,
    pub version: Version,
    id: BufferId,
    next_replica_id: Option<ReplicaId>,
    local_clock: LocalTimestamp,
    lamport_clock: LamportTimeStamp,
    fragments: Tree<Fragment>,
    insertion_splits: HashMap<EditId, Tree<InsertionSplit>>,
    undo_stack: Vec<Version>,
}

pub struct Iter<'a> {
    version: &'a Version,
    fragment_cursor: tree::Cursor<'a, Fragment>,
    fragment_offset: usize,
}

pub struct BackwardIter<'a> {
    fragment_cursor: tree::Cursor<'a, Fragment>,
    fragment_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version(Arc<HashMap<ReplicaId, LocalTimestamp>>);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FragmentId(Arc<Vec<u16>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fragment {
    id: FragmentId,
    insertion: Insertion,
    start_offset: usize,
    end_offset: usize,
    deletions: HashSet<EditId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentSummary {
    extent: usize,
    extent_2d: Point,
    max_fragment_id: FragmentId,
    first_row_len: u32,
    longest_row: u32,
    longest_row_len: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertionSplit {
    extent: usize,
    fragment_id: FragmentId,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InsertionSplitSummary {
    extent: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EditId {
    replica_id: ReplicaId,
    timestamp: LocalTimestamp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Insertion {
    id: EditId,
    parent_id: EditId,
    offset_in_parent: usize,
    replica_id: ReplicaId,
    text: Arc<Text>,
    timestamp: LamportTimeStamp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Deletion {
    start_id: EditId,
    start_offset: usize,
    end_id: EditId,
    end_offset: usize,
    version_in_range: Version,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Text {
    text: String,
    nodes: Vec<LineNode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineNode {
    len: u32,
    longest_row: u32,
    longest_row_len: u32,
    offset: usize,
    rows: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineNodeProbe<'a> {
    offset_range: &'a Range<usize>,
    row: u32,
    left_ancestor_end_offset: usize,
    right_ancestor_start_offset: usize,
    node: &'a LineNode,
    left_child: Option<&'a LineNode>,
    right_child: Option<&'a LineNode>,
}

#[derive(Debug)]
pub enum Operation {
    Edit {
        id: EditId,
        start_id: EditId,
        start_offset: usize,
        end_id: EditId,
        end_offset: usize,
        version_in_range: Version,
        timestamp: LamportTimeStamp,
        new_text: Option<Arc<Text>>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point {
    pub row: u32,
    pub col: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CharCount(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InsertionOffset(pub usize);

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    OffsetOutOfRange,
    InvalidOperation,
}

impl Buffer {
    pub fn new(id: BufferId) -> Buffer {
        let mut fragments = Tree::new();
        let sentinel_id = EditId {
            replica_id: 0,
            timestamp: 0,
        };

        fragments.push(Fragment::new(
            FragmentId::min_value(),
            Insertion {
                id: sentinel_id,
                parent_id: EditId {
                    replica_id: 0,
                    timestamp: 0,
                },
                offset_in_parent: 0,
                replica_id: 0,
                text: Arc::new(Text::new(String::new())),
                timestamp: 0,
            },
        ));

        let mut insertion_splits = HashMap::new();

        insertion_splits.insert(
            sentinel_id,
            Tree::from_item(InsertionSplit {
                fragment_id: FragmentId::min_value(),
                extent: 0,
            }),
        );

        let version = Version::new();

        Self {
            id,
            replica_id: 1,
            version: version.clone(),
            next_replica_id: Some(2),
            local_clock: 0,
            lamport_clock: 0,
            fragments,
            insertion_splits,
            undo_stack: vec![version],
        }
    }

    pub fn id(&self) -> BufferId {
        self.id
    }

    pub fn len(&self) -> usize {
        self.fragments.len::<CharCount>().0
    }

    pub fn len_for_row(&self, row: u32) -> Result<u32, Error> {
        let row_start_offset = self.offset_for_point(Point::new(row, 0))?;
        let row_end_offset = if row >= self.max_point().row {
            self.len()
        } else {
            self.offset_for_point(Point::new(row + 1, 0))? - 1
        };

        Ok((row_end_offset - row_start_offset) as u32)
    }

    pub fn longest_row(&self) -> u32 {
        self.fragments.summary().longest_row
    }

    pub fn max_point(&self) -> Point {
        self.fragments.len::<Point>()
    }

    pub fn line(&self, row: u32) -> Result<String, Error> {
        let mut iterator = self.iter_starting_at_point(Point::new(row, 0)).peekable();

        if iterator.peek().is_none() {
            Err(Error::OffsetOutOfRange)
        } else {
            Ok(String::from_utf8(iterator.take_while(|c| *c != b'\n').collect()).unwrap())
        }
    }

    #[cfg(test)]
    fn to_string(&self) -> String {
        String::from_utf8(self.iter().collect()).unwrap()
    }

    pub fn iter(&self) -> Iter {
        Iter::new(self)
    }

    pub fn iter_starting_at_point(&self, point: Point) -> Iter {
        Iter::starting_at_point(self, point)
    }

    pub fn backward_iter_starting_at_point(&self, point: Point) -> BackwardIter {
        BackwardIter::starting_at_point(self, point)
    }

    pub fn edit<'a, I, T>(&mut self, old_ranges: I, new_text: T) -> Vec<Arc<Operation>>
    where
        I: IntoIterator<Item = &'a Range<usize>>,
        T: Into<Text>,
    {
        let new_text = new_text.into();
        let new_text = if new_text.len() > 0 {
            Some(Arc::new(new_text))
        } else {
            None
        };

        let ops = self.splice_fragments(
            old_ranges.into_iter().filter(|old_range| new_text.is_some() || old_range.end > old_range.start),
            new_text.clone(),
        );

        self.version.set(self.replica_id, self.local_clock);
        self.undo_stack.push(self.version.clone());

        ops
    }

    pub fn undo(&mut self, n: usize) {
        let n = self.undo_stack.len() - 1 - n;

        self.version = self.undo_stack[n].clone();
    }

    fn offset_for_point(&self, point: Point) -> Result<usize, Error> {
        let mut fragments_cursor = self.fragments.cursor();

        fragments_cursor.seek(&point, tree::SeekBias::Left);
        fragments_cursor
            .item()
            .ok_or(Error::OffsetOutOfRange)
            .map(|fragment| {
                let overshoot = fragment
                    .offset_for_point(point - &fragments_cursor.start::<Point>())
                    .unwrap();

                let offset = &fragments_cursor.start::<CharCount>().0 + &overshoot;

                offset
            })
    }

    fn integrate_op(&mut self, op: Arc<Operation>) -> Result<(), Error> {
        match op.as_ref() {
            Operation::Edit {
                id,
                start_id,
                start_offset,
                end_id,
                end_offset,
                new_text,
                version_in_range,
                timestamp,
            } => self.integrate_edit(
                *id,
                *start_id,
                *start_offset,
                *end_id,
                *end_offset,
                new_text.as_ref().cloned(),
                version_in_range,
                *timestamp,
            )?,
        }

        Ok(())
    }

    fn integrate_edit(
        &mut self,
        id: EditId,
        start_id: EditId,
        start_offset: usize,
        end_id: EditId,
        end_offset: usize,
        new_text: Option<Arc<Text>>,
        version_in_range: &Version,
        timestamp: LamportTimeStamp,
    ) -> Result<(), Error> {
        let mut new_text = new_text.as_ref().cloned();
        let start_fragment_id = self.resolve_fragment_id(start_id, start_offset)?;
        let end_fragment_id = self.resolve_fragment_id(end_id, end_offset)?;
        let old_fragments = self.fragments.clone();
        let mut cursor = old_fragments.cursor();
        let mut new_fragments = cursor.slice(&start_fragment_id, tree::SeekBias::Left);

        if start_offset == cursor.item().unwrap().end_offset {
            new_fragments.push(cursor.item().unwrap().clone());
            cursor.next();
        }

        while let Some(fragment) = cursor.item() {
            if new_text.is_none() && fragment.id > end_fragment_id {
                break;
            }

            if fragment.id == start_fragment_id || fragment.id == end_fragment_id {
                let split_start = if start_fragment_id == fragment.id {
                    start_offset
                } else {
                    fragment.start_offset
                };

                let split_end = if end_fragment_id == fragment.id {
                    end_offset
                } else {
                    fragment.end_offset
                };

                let (before_range, within_range, after_range) = self.split_fragment(
                    cursor.prev_item().unwrap(),
                    fragment,
                    split_start..split_end
                );

                let insertion = new_text.take().map(|new_text| {
                    self.build_fragment_to_insert(
                        id,
                        before_range.as_ref().or(cursor.prev_item()).unwrap(),
                        within_range.as_ref().or(after_range.as_ref()),
                        new_text,
                        timestamp,
                    )
                });

                if let Some(fragment) = before_range {
                    new_fragments.push(fragment);
                }

                if let Some(fragment) = insertion {
                    new_fragments.push(fragment);
                }

                if let Some(mut fragment) = within_range {
                    if version_in_range.includes(&fragment.insertion) {
                        fragment.deletions.insert(id);
                    }

                    new_fragments.push(fragment);
                }

                if let Some(fragment) = after_range {
                    new_fragments.push(fragment);
                }
            } else {
                if new_text.is_some() && should_insert_before(&fragment.insertion, timestamp, id.replica_id) {
                    new_fragments.push(self.build_fragment_to_insert(
                        id,
                        cursor.prev_item().unwrap(),
                        Some(fragment),
                        new_text.take().unwrap(),
                        timestamp,
                    ));
                }

                let mut fragment = fragment.clone();

                if version_in_range.includes(&fragment.insertion) {
                    fragment.deletions.insert(id);
                }

                new_fragments.push(fragment);
            }

            cursor.next();
        }

        if let Some(new_text) = new_text {
            new_fragments.push(self.build_fragment_to_insert(
                id,
                cursor.prev_item().unwrap(),
                None,
                new_text,
                timestamp,
            ));
        }

        new_fragments.push_tree(cursor.slice(&old_fragments.len::<CharCount>(), tree::SeekBias::Right));
        self.fragments = new_fragments;
        self.lamport_clock = std::cmp::max(self.lamport_clock, timestamp) + 1;

        Ok(())
    }

    fn resolve_fragment_id(&self, edit_id: EditId, offset: usize) -> Result<FragmentId, Error> {
        let split_tree = self.insertion_splits
            .get(&edit_id)
            .ok_or(Error::InvalidOperation)?;

        let mut cursor = split_tree.cursor();

        cursor.seek(&InsertionOffset(offset), tree::SeekBias::Left);

        Ok(cursor
            .item()
            .ok_or(Error::InvalidOperation)?
            .fragment_id
            .clone()
        )
    }

    fn splice_fragments<'a, I>(
        &mut self,
        mut old_ranges: I,
        new_text: Option<Arc<Text>>,
    ) -> Vec<Arc<Operation>>
    where
        I: Iterator<Item = &'a Range<usize>>,
    {
        let mut cur_range = old_ranges.next();

        if cur_range.is_none() {
            return Vec::new();
        }

        let replica_id = self.replica_id;
        let mut ops = Vec::with_capacity(old_ranges.size_hint().0);
        let old_fragments = self.fragments.clone();
        let mut cursor = old_fragments.cursor();
        let mut new_fragments = Tree::new();

        new_fragments.push_tree(cursor.slice(
            &CharCount(cur_range.as_ref().unwrap().start),
            tree::SeekBias::Right,
        ));

        self.local_clock += 1;
        self.lamport_clock += 1;

        let mut start_id = None;
        let mut start_offset = None;
        let mut end_id = None;
        let mut end_offset = None;
        let mut version_in_range = Version::new();

        while cur_range.is_some() && cursor.item().is_some() {
            let mut fragment = cursor.item().unwrap().clone();
            let mut fragment_start = cursor.start::<CharCount>().0;
            let mut fragment_end = fragment_start + fragment.len();
            let old_split_tree = self.insertion_splits.remove(&fragment.insertion.id).unwrap();
            let mut splits_cursor = old_split_tree.cursor();
            let mut new_split_tree = splits_cursor.slice(&InsertionOffset(fragment.start_offset), tree::SeekBias::Right);
            
            while cur_range.map_or(false, |range| range.start < fragment_end) {
                let range = cur_range.clone().unwrap();

                if range.start > fragment_start {
                    let mut prefix = fragment.clone();

                    prefix.end_offset = prefix.start_offset + (range.start - fragment_start);
                    prefix.id = FragmentId::between(&new_fragments.last().unwrap().id, &fragment.id);
                    fragment.start_offset = prefix.end_offset;
                    new_fragments.push(prefix.clone());
                    new_split_tree.push(InsertionSplit {
                        extent: prefix.end_offset - prefix.start_offset,
                        fragment_id: prefix.id,
                    });

                    fragment_start = range.start;
                }

                if range.end == fragment_start {
                    end_id = Some(new_fragments.last().unwrap().insertion.id);
                    end_offset = Some(new_fragments.last().unwrap().end_offset);
                } else if range.end == fragment_end {
                    end_id = Some(fragment.insertion.id);
                    end_offset = Some(fragment.end_offset);
                }

                if range.start == fragment_start {
                    let local_timestamp = self.local_clock;
                    let lamport_timestamp = self.lamport_clock;

                    start_id = Some(new_fragments.last().unwrap().insertion.id);
                    start_offset = Some(new_fragments.last().unwrap().end_offset);

                    if let Some(new_text) = new_text.clone() {
                        let new_fragment = self.build_fragment_to_insert(
                            EditId {
                                replica_id,
                                timestamp: local_timestamp,
                            },
                            new_fragments.last().unwrap(),
                            Some(&fragment),
                            new_text,
                            lamport_timestamp,
                        );

                        new_fragments.push(new_fragment);
                    }
                }

                if range.end < fragment_end {
                    if range.end > fragment_start {
                        let mut prefix = fragment.clone();

                        prefix.end_offset = prefix.start_offset + (range.end - fragment_start);
                        prefix.id = FragmentId::between(&new_fragments.last().unwrap().id, &fragment.id);

                        if fragment.is_visible() {
                            prefix.deletions.insert(EditId {
                                replica_id,
                                timestamp: self.local_clock,
                            });
                        }

                        fragment.start_offset = prefix.end_offset;
                        new_fragments.push(prefix.clone());
                        new_split_tree.push(InsertionSplit {
                            extent: prefix.end_offset - prefix.start_offset,
                            fragment_id: prefix.id,
                        });

                        fragment_start = range.end;
                        end_id = Some(fragment.insertion.id);
                        end_offset = Some(fragment.start_offset);
                        version_in_range.include(&fragment.insertion);
                    }
                } else {
                    version_in_range.include(&fragment.insertion);

                    if fragment.is_visible() {
                        fragment.deletions.insert(EditId {
                            replica_id,
                            timestamp: self.local_clock,
                        });
                    }
                }

                if range.end <= fragment_end {
                    ops.push(Arc::new(Operation::Edit {
                        id: EditId {
                            replica_id,
                            timestamp: self.local_clock,
                        },
                        start_id: start_id.unwrap(),
                        start_offset: start_offset.unwrap(),
                        end_id: end_id.unwrap(),
                        end_offset: end_offset.unwrap(),
                        new_text: new_text.clone(),
                        timestamp: self.lamport_clock,
                        version_in_range,
                    }));

                    start_id = None;
                    start_offset = None;
                    end_id = None;
                    end_offset = None;
                    version_in_range = Version::new();
                    cur_range = old_ranges.next();

                    if cur_range.is_some() {
                        self.local_clock += 1;
                        self.lamport_clock += 1;
                    }
                } else {
                    break;
                }
            }

            new_split_tree.push(InsertionSplit {
                extent: fragment.end_offset - fragment.start_offset,
                fragment_id: fragment.id.clone(),
            });

            splits_cursor.next();

            new_split_tree.push_tree(
                splits_cursor.slice(&old_split_tree.len::<InsertionOffset>(), tree::SeekBias::Right),
            );

            self.insertion_splits.insert(fragment.insertion.id, new_split_tree);
            new_fragments.push(fragment);
            cursor.next();

            if let Some(range) = cur_range.clone() {
                while let Some(mut fragment) = cursor.item().cloned() {
                    fragment_start = cursor.start::<CharCount>().0;
                    fragment_end = fragment_start + fragment.len();

                    if range.start < fragment_start && range.end >= fragment_end {
                        if fragment.is_visible() {
                            fragment.deletions.insert(EditId {
                                replica_id,
                                timestamp: self.local_clock,
                            });
                        }

                        version_in_range.include(&fragment.insertion);
                        new_fragments.push(fragment.clone());
                        cursor.next();

                        if range.end == fragment_end {
                            end_id = Some(fragment.insertion.id);
                            end_offset = Some(fragment.end_offset);

                            ops.push(Arc::new(Operation::Edit {
                                id: EditId {
                                    replica_id,
                                    timestamp: self.local_clock,
                                },
                                start_id: start_id.unwrap(),
                                start_offset: start_offset.unwrap(),
                                end_id: end_id.unwrap(),
                                end_offset: end_offset.unwrap(),
                                new_text: new_text.clone(),
                                timestamp: self.lamport_clock,
                                version_in_range,
                            }));

                            start_id = None;
                            start_offset = None;
                            end_id = None;
                            end_offset = None;
                            version_in_range = Version::new();
                            cur_range = old_ranges.next();

                            if cur_range.is_some() {
                                self.local_clock += 1;
                                self.lamport_clock += 1;
                            }

                            break;
                        }
                    } else {
                        break;
                    }
                }

                if cur_range.map_or(false, |range| range.start > fragment_end) {
                    new_fragments.push_tree(cursor.slice(
                        &CharCount(cur_range.as_ref().unwrap().start),
                        tree::SeekBias::Right,
                    ));
                }
            }
        }

        if cur_range.is_some() {
            let local_timestamp = self.local_clock;
            let lamport_timestamp = self.lamport_clock;
            let id = EditId {
                replica_id,
                timestamp: local_timestamp,
            };

            ops.push(Arc::new(Operation::Edit {
                id,
                start_id: new_fragments.last().unwrap().insertion.id,
                start_offset: new_fragments.last().unwrap().end_offset,
                end_id: new_fragments.last().unwrap().insertion.id,
                end_offset: new_fragments.last().unwrap().end_offset,
                new_text: new_text.clone(),
                timestamp: lamport_timestamp,
                version_in_range: Version::new(),
            }));

            if let Some(new_text) = new_text {
                let new_fragment = self.build_fragment_to_insert(
                    id,
                    new_fragments.last().unwrap(),
                    None,
                    new_text,
                    lamport_timestamp,
                );

                new_fragments.push(new_fragment);
            }
        } else {
            new_fragments.push_tree(cursor.slice(&old_fragments.len::<CharCount>(), tree::SeekBias::Right));
        }

        self.fragments = new_fragments;
        ops
    }

    fn split_fragment(
        &mut self,
        prev_fragment: &Fragment,
        fragment: &Fragment,
        range: Range<usize>,
    ) -> (Option<Fragment>, Option<Fragment>, Option<Fragment>) {
        if range.end == fragment.start_offset {
            (None, None, Some(fragment.clone()))
        } else if range.start == fragment.end_offset {
            (Some(fragment.clone()), None, None)
        } else if range.start == fragment.start_offset && range.end == fragment.end_offset {
            (None, Some(fragment.clone()), None)
        } else {
            let mut prefix = fragment.clone();
            let after_range = if range.end < fragment.end_offset {
                let mut suffix = prefix.clone();

                suffix.start_offset = range.end;
                prefix.end_offset = range.end;
                prefix.id = FragmentId::between(&prev_fragment.id, &suffix.id);

                Some(suffix)
            } else {
                None
            };

            let within_range = if range.start != range.end {
                let mut suffix = prefix.clone();

                suffix.start_offset = range.start;
                prefix.end_offset = range.start;
                prefix.id = FragmentId::between(&prev_fragment.id, &suffix.id);

                Some(suffix)
            } else {
                None
            };

            let before_range = if range.start > fragment.start_offset {
                Some(prefix)
            } else {
                None
            };

            let old_split_tree = self.insertion_splits.remove(&fragment.insertion.id).unwrap();
            let mut cursor = old_split_tree.cursor();
            let mut new_split_tree = cursor.slice(&InsertionOffset(fragment.start_offset), tree::SeekBias::Right);

            if let Some(ref fragment) = before_range {
                new_split_tree.push(InsertionSplit {
                    extent: range.start - fragment.start_offset,
                    fragment_id: fragment.id.clone(),
                });
            }

            if let Some(ref fragment) = within_range {
                new_split_tree.push(InsertionSplit {
                    extent: range.end - range.start,
                    fragment_id: fragment.id.clone(),
                });
            }

            if let Some(ref fragment) = after_range {
                new_split_tree.push(InsertionSplit {
                    extent: fragment.end_offset - range.end,
                    fragment_id: fragment.id.clone(),
                });
            }

            cursor.next();
            new_split_tree.push_tree(cursor.slice(&old_split_tree.len::<InsertionOffset>(), tree::SeekBias::Right));

            self.insertion_splits.insert(fragment.insertion.id, new_split_tree);

            (before_range, within_range, after_range)
        }
    }

    fn build_fragment_to_insert(
        &mut self,
        edit_id: EditId,
        prev_fragment: &Fragment,
        next_fragment: Option<&Fragment>,
        text: Arc<Text>,
        timestamp: LamportTimeStamp,
    ) -> Fragment {
        let new_fragment_id = FragmentId::between(
            &prev_fragment.id,
            next_fragment.map(|f| &f.id).unwrap_or(&FragmentId::max_value()),
        );

        let mut split_tree = Tree::new();

        split_tree.push(InsertionSplit {
            extent: text.len(),
            fragment_id: new_fragment_id.clone(),
        });

        self.insertion_splits.insert(edit_id, split_tree);

        Fragment::new(
            new_fragment_id,
            Insertion {
                id: edit_id,
                parent_id: prev_fragment.insertion.id,
                offset_in_parent: prev_fragment.end_offset,
                replica_id: self.replica_id,
                text,
                timestamp,
            },
        )
    }
}

fn should_insert_before(
    insertion: &Insertion,
    other_timestamp: LamportTimeStamp,
    other_replica_id: ReplicaId,
) -> bool {
    match insertion.timestamp.cmp(&other_timestamp) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Equal => insertion.id.replica_id < other_replica_id,
        std::cmp::Ordering::Greater => false,
    }
}

impl Version {
    fn new() -> Version {
        Version(Arc::new(HashMap::new()))
    }

    fn set(&mut self, replica_id: ReplicaId, timestamp: LocalTimestamp) {
        let map = Arc::make_mut(&mut self.0);

        *map.entry(replica_id).or_insert(0) = timestamp;
    }

    fn include(&mut self, insertion: &Insertion) {
        let map = Arc::make_mut(&mut self.0);
        let value = map.entry(insertion.id.replica_id).or_insert(0);

        *value = std::cmp::max(*value, insertion.id.timestamp);
    }

    fn includes(&self, insertion: &Insertion) -> bool {
        if let Some(timestamp) = self.0.get(&insertion.id.replica_id) {
            *timestamp >= insertion.id.timestamp
        } else {
            false
        }
    }
}

impl<'a> Iter<'a> {
    fn new(buffer: &'a Buffer) -> Iter<'a> {
        let mut fragment_cursor = buffer.fragments.cursor();

        fragment_cursor.seek(&CharCount(0), tree::SeekBias::Right);

        Iter {
            version: &buffer.version,
            fragment_cursor,
            fragment_offset: 0,
        }
    }

    fn starting_at_point(buffer: &'a Buffer, point: Point) -> Iter<'a> {
        let mut fragment_cursor = buffer.fragments.cursor();

        fragment_cursor.seek(&point, tree::SeekBias::Right);

        let fragment_offset = if let Some(fragment) = fragment_cursor.item() {
            let point_in_fragment = point - &fragment_cursor.start::<Point>();

            fragment.offset_for_point(point_in_fragment).unwrap()
        } else {
            0
        };

        Iter {
            version: &buffer.version,
            fragment_cursor,
            fragment_offset,
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if let Some(fragment) = self.fragment_cursor.item() {
            if self.version.includes(&fragment.insertion) {
                if let Some(c) = fragment.get_byte(self.fragment_offset) {
                    self.fragment_offset += 1;

                    return Some(c);
                }
            }
        }

        loop {
            self.fragment_cursor.next();

            if let Some(fragment) = self.fragment_cursor.item() {
                if self.version.includes(&fragment.insertion) {
                    if let Some(c) = fragment.get_byte(0) {
                        self.fragment_offset = 1;

                        return Some(c);
                    }
                }
            } else {
                break;
            }
        }

        None
    }
}

impl<'a> BackwardIter<'a> {
    fn starting_at_point(buffer: &'a Buffer, point: Point) -> BackwardIter<'a> {
        let mut fragment_cursor = buffer.fragments.cursor();

        fragment_cursor.seek(&point, tree::SeekBias::Left);

        let fragment_offset = if let Some(fragment) = fragment_cursor.item() {
            let point_in_fragment = point - &fragment_cursor.start::<Point>();

            fragment.offset_for_point(point_in_fragment).unwrap()
        } else {
            0
        };

        BackwardIter {
            fragment_cursor,
            fragment_offset,
        }
    }
}

impl<'a> Iterator for BackwardIter<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if let Some(fragment) = self.fragment_cursor.item() {
            if self.fragment_offset > 0 {
                self.fragment_offset -= 1;

                if let Some(c) = fragment.get_byte(self.fragment_offset) {
                    return Some(c);
                }
            }
        }

        loop {
            self.fragment_cursor.prev();

            if let Some(fragment) = self.fragment_cursor.item() {
                if fragment.len() > 0 {
                    self.fragment_offset = fragment.len() - 1;

                    return fragment.get_byte(self.fragment_offset);
                }
            } else {
                break;
            }
        }

        None
    }
}

impl Text {
    fn new(text: String) -> Text {
        fn build_tree(index: usize, line_lengths: &[u32], tree: &mut [LineNode]) {
            if line_lengths.is_empty() {
                return;
            }

            let mid = if line_lengths.len() == 1 {
                0
            } else {
                let depth = log2_fast(line_lengths.len());
                let max_elements = (1 << depth) - 1;
                let right_subtree_elements = 1 << (depth - 1);

                std::cmp::min(line_lengths.len() - right_subtree_elements, max_elements)
            };

            let len = line_lengths[mid];
            let lower = &line_lengths[0..mid];
            let upper = &line_lengths[mid + 1..];
            let left_child_index = index * 2 + 1;
            let right_child_index = index * 2 + 2;

            build_tree(left_child_index, lower, tree);
            build_tree(right_child_index, upper, tree);

            tree[index] = {
                let mut left_child_longest_row = 0;
                let mut left_child_longest_row_len = 0;
                let mut left_child_offset = 0;
                let mut left_child_rows = 0;

                if let Some(left_child) = tree.get(left_child_index) {
                    left_child_longest_row = left_child.longest_row;
                    left_child_longest_row_len = left_child.longest_row_len;
                    left_child_offset = left_child.offset;
                    left_child_rows = left_child.rows;
                }

                let mut right_child_longest_row = 0;
                let mut right_child_longest_row_len = 0;
                let mut right_child_offset = 0;
                let mut right_child_rows = 0;

                if let Some(right_child) = tree.get(right_child_index) {
                    right_child_longest_row = right_child.longest_row;
                    right_child_longest_row_len = right_child.longest_row_len;
                    right_child_offset = right_child.offset;
                    right_child_rows = right_child.rows;
                }

                let mut longest_row = 0;
                let mut longest_row_len = 0;

                if left_child_longest_row_len > longest_row_len {
                    longest_row = left_child_longest_row;
                    longest_row_len = left_child_longest_row_len;
                }

                if len > longest_row_len {
                    longest_row = left_child_rows;
                    longest_row_len = len;
                }

                if right_child_longest_row_len > longest_row_len {
                    longest_row = left_child_rows - right_child_longest_row + 1;
                    longest_row_len = right_child_longest_row_len;
                }

                LineNode {
                    len,
                    longest_row,
                    longest_row_len,
                    offset: left_child_offset + len as usize + right_child_offset + 1,
                    rows: left_child_rows + right_child_rows + 1,
                }
            };
        }

        let mut line_lengths = Vec::new();
        let mut prev_offset = 0;

        for (offset, byte) in text.bytes().enumerate() {
            if byte == b'\n' {
                line_lengths.push((offset - prev_offset) as u32);
                prev_offset = offset + 1;
            }
        }

        line_lengths.push((text.len() - prev_offset) as u32);

        let mut nodes = Vec::new();

        nodes.resize(
            line_lengths.len(),
            LineNode {
                len: 0,
                longest_row_len: 0,
                longest_row: 0,
                offset: 0,
                rows: 0,
            },
        );

        build_tree(0, &line_lengths, &mut nodes);

        Text {
            text,
            nodes
        }
    }

    fn len(&self) -> usize {
        self.text.len()
    }

    fn longest_row_in_range(&self, target_range: Range<usize>) -> Result<(u32, u32), Error> {
        let mut longest_row = 0;
        let mut longest_row_len = 0;

        self.search(|probe| {
            if target_range.start <= probe.offset_range.end && probe.right_ancestor_start_offset <= target_range.end {
                if let Some(right_child) = probe.right_child {
                    longest_row = probe.row + 1 + right_child.longest_row;
                    longest_row_len = right_child.longest_row_len;
                }
            }
            
            if target_range.start < probe.offset_range.start {
                if probe.offset_range.end < target_range.end && probe.node.len >= longest_row_len {
                    longest_row = probe.row;
                    longest_row_len = probe.node.len;
                }

                std::cmp::Ordering::Less
            } else if target_range.start > probe.offset_range.end {
                std::cmp::Ordering::Greater
            } else {
                let node_end = std::cmp::min(probe.offset_range.end, target_range.end);
                let node_len = (node_end - target_range.start) as u32;

                if node_len >= longest_row_len {
                    longest_row = probe.row;
                    longest_row_len = node_len;
                }

                std::cmp::Ordering::Equal
            }
        }).ok_or(Error::OffsetOutOfRange)?;

        self.search(|probe| {
            if target_range.end >= probe.offset_range.start && probe.left_ancestor_end_offset >= target_range.start {
                if let Some(left_child) = probe.left_child {
                    if left_child.longest_row_len > longest_row_len {
                        let left_ancestor_row = probe.row - left_child.rows;

                        longest_row = left_ancestor_row + left_child.longest_row;
                        longest_row_len = left_child.longest_row_len;
                    }
                }
            }

            if target_range.end < probe.offset_range.start {
                std::cmp::Ordering::Less
            } else if target_range.end > probe.offset_range.end {
                if target_range.start < probe.offset_range.start && probe.node.len > longest_row_len {
                    longest_row = probe.row;
                    longest_row_len = probe.node.len;
                }

                std::cmp::Ordering::Greater
            } else {
                let node_start = std::cmp::max(target_range.start, probe.offset_range.start);
                let node_len = (target_range.end - node_start) as u32;

                if node_len > longest_row_len {
                    longest_row = probe.row;
                    longest_row_len = node_len;
                }

                std::cmp::Ordering::Equal
            }
        }).ok_or(Error::OffsetOutOfRange)?;

        Ok((longest_row, longest_row_len))
    }

    fn point_for_offset(&self, offset: usize) -> Result<Point, Error> {
        let search_result = self.search(|probe| {
            if offset < probe.offset_range.start {
                std::cmp::Ordering::Less
            } else if offset > probe.offset_range.end {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });

        if let Some((offset_range, row, _)) = search_result {
            Ok(Point::new(row, (offset - offset_range.start) as u32))
        } else {
            Err(Error::OffsetOutOfRange)
        }
    }

    fn offset_for_point(&self, point: Point) -> Result<usize, Error> {
        if let Some((offset_range, _, node)) = self.search(|probe| point.row.cmp(&probe.row)) {
            if point.col <= node.len {
                Ok(offset_range.start + point.col as usize)
            } else {
                Err(Error::OffsetOutOfRange)
            }
        } else {
            Err(Error::OffsetOutOfRange)
        }
    }

    fn search<F>(&self, mut f: F) -> Option<(Range<usize>, u32, &LineNode)>
    where
        F: FnMut(LineNodeProbe) -> std::cmp::Ordering,
    {
        let mut left_ancestor_end_offset = 0;
        let mut left_ancestor_row = 0;
        let mut right_ancestor_start_offset = self.nodes[0].offset;
        let mut cur_node_index = 0;

        while let Some(cur_node) = self.nodes.get(cur_node_index) {
            let left_child = self.nodes.get(cur_node_index * 2 + 1);
            let right_child = self.nodes.get(cur_node_index * 2 + 2);
            let cur_offset_range = {
                let start = left_ancestor_end_offset + left_child.map_or(0, |node| node.offset);
                let end = start + cur_node.len as usize;

                start..end
            };

            let cur_row = left_ancestor_row + left_child.map_or(0, |node| node.rows);

            match f(LineNodeProbe {
                offset_range: &cur_offset_range,
                row: cur_row,
                left_ancestor_end_offset,
                right_ancestor_start_offset,
                node: cur_node,
                left_child,
                right_child,
            }) {
                std::cmp::Ordering::Less => {
                    cur_node_index = cur_node_index * 2 + 1;
                    right_ancestor_start_offset = cur_offset_range.start;
                },
                std::cmp::Ordering::Equal => return Some((cur_offset_range, cur_row, cur_node)),
                std::cmp::Ordering::Greater => {
                    cur_node_index = cur_node_index * 2 + 1;
                    left_ancestor_end_offset = cur_offset_range.end + 1;
                    left_ancestor_row = cur_row + 1;
                },
            }
        }

        None
    }
}

impl<'a> From<&'a str> for Text {
    fn from(src: &'a str) -> Text {
        Text::new(src.to_string())
    }
}

impl From<String> for Text {
    fn from(src: String) -> Text {
        Text::new(src)
    }
}

#[inline(always)]
fn log2_fast(x: usize) -> usize {
    8 * std::mem::size_of::<usize>() - (x.leading_zeros() as usize) - 1
}

impl FragmentId {
    fn min_value() -> FragmentId {
        FragmentId(Arc::new(vec![0u16]))
    }

    fn max_value() -> FragmentId {
        FragmentId(Arc::new(vec![u16::max_value()]))
    }

    fn between(left: &FragmentId, right: &FragmentId) -> FragmentId {
        FragmentId::between_with_max(left, right, u16::max_value())
    }

    fn between_with_max(left: &FragmentId, right: &FragmentId, max_value: u16) -> FragmentId {
        let mut new_entries = Vec::new();
        let left_entries = left.0.iter().cloned().chain(std::iter::repeat(0));
        let right_entries = right.0.iter().cloned().chain(std::iter::repeat(max_value));

        for (l, r) in left_entries.zip(right_entries) {
            let interval = r - l;

            if interval > 1 {
                new_entries.push(l + interval / 2);

                break;
            } else {
                new_entries.push(l);
            }
        }

        FragmentId(Arc::new(new_entries))
    }
}

impl tree::Dimension for FragmentId {
    type Summary = FragmentSummary;

    fn from_summary(summary: &FragmentSummary) -> FragmentId {
        summary.max_fragment_id.clone()
    }
}

impl<'a> Add<&'a FragmentId> for FragmentId {
    type Output = FragmentId;

    fn add(self, other: &FragmentId) -> FragmentId {
        std::cmp::max(&self, other).clone()
    }
}

impl Fragment {
    fn new(id: FragmentId, insertion: Insertion) -> Fragment {
        let end_offset = insertion.text.len();

        Fragment {
            id,
            insertion,
            start_offset: 0,
            end_offset,
            deletions: HashSet::new(),
        }
    }

    fn get_byte(&self, offset: usize) -> Option<u8> {
        if offset < self.len() {
            Some(self.insertion.text.text.as_bytes()[self.start_offset + offset])
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        if self.is_visible() {
            self.end_offset - self.start_offset
        } else {
            0
        }
    }

    fn is_visible(&self) -> bool {
        self.deletions.is_empty()
    }

    fn point_for_offset(&self, offset: usize) -> Result<Point, Error> {
        let text = &self.insertion.text;
        let offset_in_intersection = self.start_offset + offset;

        Ok(text.point_for_offset(offset_in_intersection)? - &text.point_for_offset(self.start_offset)?)
    }

    fn offset_for_point(&self, point: Point) -> Result<usize, Error> {
        let text = &self.insertion.text;
        let point_in_intersection = text.point_for_offset(self.start_offset)? + &point;

        Ok(text.offset_for_point(point_in_intersection)? - self.start_offset)
    }
}

impl tree::Item for Fragment {
    type Summary = FragmentSummary;

    fn summarize(&self) -> FragmentSummary {
        if self.is_visible() {
            let fragment_2d_start = self.insertion
                .text
                .point_for_offset(self.start_offset)
                .unwrap();

            let fragment_2d_end = self.insertion
                .text
                .point_for_offset(self.end_offset)
                .unwrap();

            let first_row_len = if fragment_2d_start.row == fragment_2d_end.row {
                (self.end_offset - self.start_offset) as u32
            } else {
                let first_row_end = self.insertion
                    .text
                    .offset_for_point(Point::new(fragment_2d_start.row + 1, 0))
                    .unwrap() - 1;

                (first_row_end - self.start_offset) as u32
            };

            let (longest_row, longest_row_len) = self.insertion
                .text
                .longest_row_in_range(self.start_offset..self.end_offset)
                .unwrap();

            FragmentSummary {
                extent: self.len(),
                extent_2d: fragment_2d_end - &fragment_2d_start,
                max_fragment_id: self.id.clone(),
                first_row_len,
                longest_row: longest_row - fragment_2d_start.row,
                longest_row_len,
            }
        } else {
            FragmentSummary {
                max_fragment_id: self.id.clone(),
                .. Default::default()
            }
        }
    }
}

impl<'a> AddAssign<&'a FragmentSummary> for FragmentSummary {
    fn add_assign(&mut self, other: &FragmentSummary) {
        let last_row_len = self.extent_2d.col + other.first_row_len;

        if last_row_len > self.longest_row_len {
            self.longest_row = self.extent_2d.row;
            self.longest_row_len = last_row_len;
        }

        if other.longest_row_len > self.longest_row_len {
            self.longest_row = self.extent_2d.row + other.longest_row;
            self.longest_row_len = other.longest_row_len;
        }

        self.extent += other.extent;
        self.extent_2d.row += other.extent_2d.row;
        self.extent_2d.col += other.extent_2d.col;

        if self.max_fragment_id < other.max_fragment_id {
            self.max_fragment_id = other.max_fragment_id.clone();
        }
    }
}

impl Default for FragmentSummary {
    fn default() -> FragmentSummary {
        FragmentSummary {
            extent: 0,
            extent_2d: Point { row: 0, col: 0 },
            max_fragment_id: FragmentId::min_value(),
            first_row_len: 0,
            longest_row: 0,
            longest_row_len: 0,
        }
    }
}

impl tree::Item for InsertionSplit {
    type Summary = InsertionSplitSummary;

    fn summarize(&self) -> InsertionSplitSummary {
        InsertionSplitSummary {
            extent: self.extent,
        }
    }
}

impl<'a> AddAssign<&'a InsertionSplitSummary> for InsertionSplitSummary {
    fn add_assign(&mut self, other: &Self) {
        self.extent += other.extent;
    }
}

impl Point {
    pub fn new(row: u32, col: u32) -> Point {
        Point {
            row,
            col,
        }
    }
}

impl tree::Dimension for Point {
    type Summary = FragmentSummary;

    fn from_summary(summary: &FragmentSummary) -> Point {
        summary.extent_2d
    }
}

impl<'a> Add<&'a Point> for Point {
    type Output = Point;

    fn add(self, other: &'a Point) -> Point {
        if other.row == 0 {
            Point::new(self.row, self.col + other.col)
        } else {
            Point::new(self.row + other.row, other.col)
        }
    }
}

impl<'a> std::ops::Sub<&'a Point> for Point {
    type Output = Point;

    fn sub(self, other: &'a Point) -> Point {
        if self.row == other.row {
            Point::new(0, self.col - other.col)
        } else {
            Point::new(self.row - other.row, self.col)
        }
    }
}

impl PartialOrd for Point {
    fn partial_cmp(&self, other: &Point) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Point {
    #[cfg(target_pointer_width = "64")]
    fn cmp(&self, other: &Point) -> std::cmp::Ordering {
        let a = (self.row as usize) << 32 | self.col as usize;
        let b = (other.row as usize) << 32 | other.col as usize;

        a.cmp(&b)
    }

    #[cfg(target_pointer_width = "32")]
    fn cmp(&self, other: &Point) -> std::cmp::Ordering {
        match self.row.cmp(&other.row) {
            std::cmp::Ordering::Equal => self.column.cmp(&other.col),
            cmp @ _ => cmp,
        }
    }
}

impl tree::Dimension for CharCount {
    type Summary = FragmentSummary;

    fn from_summary(summary: &FragmentSummary) -> CharCount {
        CharCount(summary.extent)
    }
}

impl<'a> Add<&'a CharCount> for CharCount {
    type Output = CharCount;

    fn add(self, other: &'a CharCount) -> CharCount {
        CharCount(self.0 + other.0)
    }
}

impl tree::Dimension for InsertionOffset {
    type Summary = InsertionSplitSummary;

    fn from_summary(summary: &InsertionSplitSummary) -> InsertionOffset {
        InsertionOffset(summary.extent)
    }
}

impl<'a> Add<&'a InsertionOffset> for InsertionOffset {
    type Output = InsertionOffset;

    fn add(self, other: &'a InsertionOffset) -> InsertionOffset {
        InsertionOffset(self.0 + other.0)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_edit() {
        let mut buffer = Buffer::new(0);

        buffer.edit(&[0..0], "hello world");
        assert_eq!(buffer.to_string(), "hello world");
        
        buffer.edit(&[5..5], ", there");
        buffer.edit(&[18..18], "!");
        assert_eq!(buffer.to_string(), "hello, there world!");

        buffer.edit(&[16..18], "ms");
        assert_eq!(buffer.to_string(), "hello, there worms!");

        buffer.undo(1);

        panic!("{}", buffer.to_string());
    }
}
