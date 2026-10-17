"""Line counts use a track's bbox center and the finite, oriented segment."""
import numpy as np
import pytest

from src.event_engine import EventEngine
from src.schemas import Event


HORIZONTAL_LINE = [[10, 50], [90, 50]]


@pytest.mark.parametrize("before,after,direction,left,right", [
    ((50, 40), (50, 60), "left_to_right", 0, 1),
    ((50, 60), (50, 40), "right_to_left", 1, 0),
], ids=["negative-to-positive", "positive-to-negative"])
def test_crossing_counts_once_in_the_correct_direction(
        tracking_packet_factory, before, after, direction, left, right):
    engine = EventEngine(Event(line=HORIZONTAL_LINE))
    initial = engine.process(tracking_packet_factory([(1, *before)], timestamp=1))
    assert initial.events[1]["crossed"] is False
    assert initial.left_count == initial.right_count == 0

    result = engine.process(tracking_packet_factory([(1, *after)], timestamp=2))
    assert result.events[1]["crossed"] is True
    assert result.events[1]["direction"] == direction
    assert (result.left_count, result.right_count) == (left, right)
    assert result.events[1]["zone_entry"] is False

    # Staying on the destination side must not count this crossing again.
    repeated = engine.process(tracking_packet_factory([(1, *after)], timestamp=3))
    assert repeated.events[1]["crossed"] is False
    assert repeated.events[1]["direction"] is None
    assert (repeated.left_count, repeated.right_count) == (left, right)


@pytest.mark.parametrize("line,before,after", [
    ([[10, 50], [90, 50]], (50, 40), (50, 60)),
    ([[50, 90], [50, 10]], (40, 50), (60, 50)),
    ([[10, 10], [90, 90]], (60, 40), (40, 60)),
], ids=["horizontal", "vertical", "diagonal"])
def test_crossing_supports_different_line_orientations(
        tracking_packet_factory, line, before, after):
    engine = EventEngine(Event(line=line))
    engine.process(tracking_packet_factory([(7, *before)], timestamp=1))
    result = engine.process(tracking_packet_factory([(7, *after)], timestamp=2))
    assert result.events[7]["crossed"] is True
    assert result.events[7]["direction"] == "left_to_right"
    assert result.right_count == 1 and result.left_count == 0


def test_reversing_endpoints_reverses_the_counted_direction(tracking_packet_factory):
    forward = EventEngine(Event(line=HORIZONTAL_LINE))
    reverse = EventEngine(Event(line=list(reversed(HORIZONTAL_LINE))))
    for engine in (forward, reverse):
        engine.process(tracking_packet_factory([(1, 50, 40)], timestamp=1))
    packet = tracking_packet_factory([(1, 50, 60)], timestamp=2)
    a, b = forward.process(packet), reverse.process(packet)
    assert a.events[1]["direction"] == "left_to_right"
    assert b.events[1]["direction"] == "right_to_left"
    assert (a.left_count, a.right_count) == (0, 1)
    assert (b.left_count, b.right_count) == (1, 0)


@pytest.mark.parametrize("positions", [
    [(20, 40), (50, 40), (80, 40)],
    [(50, 40), (60, 45), (70, 49)],
    [(20, 50), (50, 50), (80, 50)],
], ids=["parallel-to-line", "approaches-without-crossing", "moves-along-line"])
def test_motion_without_changing_sides_does_not_count(tracking_packet_factory, positions):
    engine = EventEngine(Event(line=HORIZONTAL_LINE))
    for timestamp, (x, y) in enumerate(positions):
        result = engine.process(tracking_packet_factory([(1, x, y)], timestamp=timestamp))
        assert result.events[1]["crossed"] is False
        assert result.events[1]["direction"] is None
        assert result.left_count == result.right_count == 0


@pytest.mark.parametrize("ys,crossed,left,right", [
    ([40, 50, 50, 60], [False, False, False, True], 0, 1),
    ([40, 50, 50, 40], [False, False, False, False], 0, 0),
    ([50, 50, 60], [False, False, False], 0, 0),
], ids=["pause-on-line-then-cross", "touch-line-and-return", "first-seen-on-line"])
def test_points_on_line_preserve_the_last_nonzero_side(
        tracking_packet_factory, ys, crossed, left, right):
    engine = EventEngine(Event(line=HORIZONTAL_LINE))
    results = [engine.process(tracking_packet_factory([(1, 50, y)], timestamp=i))
               for i, y in enumerate(ys)]
    assert [result.events[1]["crossed"] for result in results] == crossed
    assert (results[-1].left_count, results[-1].right_count) == (left, right)


@pytest.mark.parametrize("x,expected", [
    (5, False), (10, True), (50, True), (90, True), (95, False),
], ids=["before-start", "start-endpoint", "middle", "end-endpoint", "after-end"])
def test_only_intersections_with_the_finite_segment_count(tracking_packet_factory, x, expected):
    engine = EventEngine(Event(line=HORIZONTAL_LINE))
    engine.process(tracking_packet_factory([(1, x, 40)], timestamp=1))
    result = engine.process(tracking_packet_factory([(1, x, 60)], timestamp=2))
    assert result.events[1]["crossed"] is expected
    assert result.right_count == int(expected)
    assert result.left_count == 0


def test_out_of_segment_side_change_does_not_hide_a_later_real_crossing(tracking_packet_factory):
    engine = EventEngine(Event(line=HORIZONTAL_LINE))
    positions = [(5, 40), (5, 60), (50, 60), (50, 40)]
    results = [engine.process(tracking_packet_factory([(1, x, y)], timestamp=i))
               for i, (x, y) in enumerate(positions)]
    assert [result.events[1]["crossed"] for result in results] == [False, False, False, True]
    assert (results[-1].left_count, results[-1].right_count) == (1, 0)


def test_another_track_on_the_opposite_side_is_not_a_crossing(tracking_packet_factory):
    engine = EventEngine(Event(line=HORIZONTAL_LINE))
    engine.process(tracking_packet_factory([(1, 50, 40)], timestamp=1))
    result = engine.process(tracking_packet_factory([(2, 50, 60)], timestamp=2))
    assert result.events[2]["crossed"] is False
    assert result.left_count == result.right_count == 0


def test_track_order_in_packet_does_not_change_independent_counts(tracking_packet_factory):
    engine = EventEngine(Event(line=HORIZONTAL_LINE))
    engine.process(tracking_packet_factory([(1, 40, 40), (2, 60, 60)], timestamp=1))
    result = engine.process(tracking_packet_factory([(2, 60, 40), (1, 40, 60)], timestamp=2))
    assert result.events[1]["direction"] == "left_to_right"
    assert result.events[2]["direction"] == "right_to_left"
    assert (result.left_count, result.right_count) == (1, 1)


def test_repeated_back_and_forth_crossings_accumulate(tracking_packet_factory):
    engine = EventEngine(Event(line=HORIZONTAL_LINE))
    results = [engine.process(tracking_packet_factory([(1, 50, y)], timestamp=i))
               for i, y in enumerate([40, 60, 40, 60, 60])]
    assert [result.events[1]["crossed"] for result in results] == [False, True, True, True, False]
    assert (results[-1].left_count, results[-1].right_count) == (1, 2)


def test_crossing_uses_bbox_center_not_its_top_left_corner(tracking_packet_factory):
    engine = EventEngine(Event(line=HORIZONTAL_LINE))
    before = tracking_packet_factory([(1, 50, 40)], timestamp=1)
    after = tracking_packet_factory([(1, 50, 60)], timestamp=2)
    before.boxes = np.array([[30, 10, 70, 70]], dtype=np.float32)  # Center y=40.
    after.boxes = np.array([[30, 20, 70, 100]], dtype=np.float32)  # Center y=60.
    engine.process(before)
    result = engine.process(after)
    assert result.events[1]["crossed"] is True
    assert result.right_count == 1


@pytest.mark.parametrize("shape", [(100, 200, 3), (240, 160, 3)])
def test_normalized_and_pixel_lines_produce_identical_events(tracking_packet_factory, shape):
    height, width = shape[:2]
    normalized = EventEngine(Event(line=[[.1, .5], [.9, .5]], normalized=True))
    pixels = EventEngine(Event(line=[[.1 * width, .5 * height], [.9 * width, .5 * height]]))
    for timestamp, y in enumerate([.4 * height, .6 * height, .4 * height]):
        packet = tracking_packet_factory([(1, .5 * width, y)], timestamp=timestamp, frame_shape=shape)
        a, b = normalized.process(packet), pixels.process(packet)
        assert a.events == b.events
        assert (a.left_count, a.right_count) == (b.left_count, b.right_count)
    assert (a.left_count, a.right_count) == (1, 1)
