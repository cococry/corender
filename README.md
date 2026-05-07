# corender
Core rendering system of ragnar


# Pipeline sketch

## 1. walk_segments
    - segment -> tile counts
    - segment -> tile touch records
    - segment -> screen-wide backdrop delta events

## 2. allocate_tiles
    tile counts -> tile segment bases

## 3. scatter_tiles
    touch records -> compact tile segment lists

## 4. backdrop prefix
    screen-wide row prefix-add of backdrop deltas

## 5. fine
    backdrop seed + tile segment list -> coverage/image
