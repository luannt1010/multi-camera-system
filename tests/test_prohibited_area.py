"""Prohibited-area rules tested directly with tracked boxes and source pixels.

No model, camera stream, event writer, or database is needed for these tests.
"""

import numpy as np
import pytest

from src.event_engine import EventEngine
from src.schemas import Event


SQUARE = [[20, 20], [80, 20], [80, 80], [20, 80]]


def assert_zone_result(result, track_id, *, inside, entry):
    """A polygon rule must never produce a line-crossing event or count."""
    assert result.events[track_id] == {
        "in_prohibit": inside,
        "zone_entry": entry,
        "crossed": False,
        "direction": None,
    }
    assert result.left_count == result.right_count == 0


@pytest.mark.parametrize(
    "center, inside",
    [((10, 50), False), ((50, 50), True)],
    ids=["first-seen-outside", "first-seen-inside"],
)
def test_first_observation_uses_box_center(
    tracking_packet_factory, center, inside
):
    engine = EventEngine(Event(polygon=SQUARE))
    packet = tracking_packet_factory([(17, *center)])

    result = engine.process(packet)

    assert_zone_result(result, 17, inside=inside, entry=inside)


@pytest.mark.parametrize(
    "center",
    [
        (20, 50), (80, 50), (50, 20), (50, 80),
        (20, 20), (80, 20), (80, 80), (20, 80),
    ],
    ids=[
        "left-edge", "right-edge", "top-edge", "bottom-edge",
        "top-left-vertex", "top-right-vertex",
        "bottom-right-vertex", "bottom-left-vertex",
    ],
)
def test_polygon_boundary_counts_as_inside(tracking_packet_factory, center):
    engine = EventEngine(Event(polygon=SQUARE))

    result = engine.process(tracking_packet_factory([(1, *center)]))

    assert_zone_result(result, 1, inside=True, entry=True)


def test_remaining_inside_emits_only_one_entry(tracking_packet_factory):
    engine = EventEngine(Event(polygon=SQUARE))
    centers = [(20, 50), (30, 50), (50, 50), (80, 50), (70, 60)]

    for timestamp, center in enumerate(centers):
        result = engine.process(tracking_packet_factory(
            [(8, *center)], timestamp=float(timestamp)
        ))
        assert_zone_result(result, 8, inside=True, entry=timestamp == 0)


def test_exit_then_reentry_emits_another_entry(tracking_packet_factory):
    engine = EventEngine(Event(polygon=SQUARE))
    positions = [50, 90, 95, 80, 60]
    expected = [(True, True), (False, False), (False, False),
                (True, True), (True, False)]

    for timestamp, (x, (inside, entry)) in enumerate(zip(positions, expected)):
        result = engine.process(tracking_packet_factory(
            [(3, x, 50)], timestamp=float(timestamp)
        ))
        assert_zone_result(result, 3, inside=inside, entry=entry)


def test_zone_state_follows_track_ids_when_box_order_changes(
    tracking_packet_factory,
):
    engine = EventEngine(Event(polygon=SQUARE))
    observations = [
        ([(10, 50, 50), (20, 10, 50)], {10: (True, True), 20: (False, False)}),
        ([(20, 50, 50), (10, 60, 50)], {20: (True, True), 10: (True, False)}),
        ([(10, 90, 50), (20, 60, 60)], {10: (False, False), 20: (True, False)}),
        ([(20, 90, 50), (10, 50, 50)], {20: (False, False), 10: (True, True)}),
    ]

    for timestamp, (objects, expected) in enumerate(observations):
        result = engine.process(tracking_packet_factory(
            objects, timestamp=float(timestamp)
        ))
        assert set(result.events) == {10, 20}
        for track_id, (inside, entry) in expected.items():
            assert_zone_result(result, track_id, inside=inside, entry=entry)


@pytest.mark.parametrize(
    "box, inside",
    [([15, 40, 23, 60], False), ([-20, -20, 120, 120], True)],
    ids=["overlap-with-center-outside", "corners-outside-with-center-inside"],
)
def test_membership_uses_center_not_box_overlap(
    tracking_packet_factory, box, inside
):
    engine = EventEngine(Event(polygon=SQUARE))
    packet = tracking_packet_factory([(1, 50, 50)])
    packet.boxes = np.asarray([box], dtype=np.float32)

    result = engine.process(packet)

    assert_zone_result(result, 1, inside=inside, entry=inside)


@pytest.mark.parametrize(
    "center, inside",
    [((60, 30), True), ((30, 60), True), ((60, 60), False)],
    ids=["horizontal-arm", "vertical-arm", "concave-notch"],
)
def test_concave_polygon_does_not_fill_its_notch(
    tracking_packet_factory, center, inside
):
    polygon = [[20, 20], [80, 20], [80, 40], [40, 40], [40, 80], [20, 80]]
    engine = EventEngine(Event(polygon=polygon))

    result = engine.process(tracking_packet_factory([(1, *center)]))

    assert_zone_result(result, 1, inside=inside, entry=inside)


def test_reversing_vertex_order_preserves_membership_and_entry(
    tracking_packet_factory,
):
    clockwise = EventEngine(Event(polygon=SQUARE))
    counterclockwise = EventEngine(Event(polygon=SQUARE[::-1]))

    for timestamp, x in enumerate([10, 20, 50, 90, 80]):
        packet = tracking_packet_factory([(4, x, 50)], timestamp=float(timestamp))
        forward = clockwise.process(packet)
        reversed_order = counterclockwise.process(packet)

        assert forward.events == reversed_order.events
        assert forward.left_count == reversed_order.left_count == 0
        assert forward.right_count == reversed_order.right_count == 0


@pytest.mark.parametrize(
    "shape", [(100, 200, 3), (180, 320, 3)],
    ids=["200x100", "320x180"],
)
def test_normalized_polygon_matches_pixel_geometry_at_source_resolution(
    tracking_packet_factory, shape
):
    normalized_polygon = [[.25, .25], [.75, .25], [.75, .75], [.25, .75]]
    height, width = shape[:2]
    pixels = (np.asarray(normalized_polygon) * [width, height]).tolist()
    normalized = EventEngine(Event(polygon=normalized_polygon, normalized=True))
    pixel = EventEngine(Event(polygon=pixels))
    x_positions = [.1, .25, .5, .9, .5]
    expected_inside = [False, True, True, False, True]
    expected_entry = [False, True, False, False, True]

    for timestamp, x in enumerate(x_positions):
        packet = tracking_packet_factory(
            [(1, x * width, .5 * height)],
            timestamp=float(timestamp), frame_shape=shape,
        )
        normalized_result = normalized.process(packet)
        pixel_result = pixel.process(packet)

        assert normalized_result.events == pixel_result.events
        assert_zone_result(
            normalized_result, 1,
            inside=expected_inside[timestamp], entry=expected_entry[timestamp],
        )


def test_line_only_rule_keeps_zone_flags_false(tracking_packet_factory):
    engine = EventEngine(Event(line=[[20, 50], [80, 50]]))
    results = [engine.process(tracking_packet_factory(
        [(1, 50, y)], timestamp=float(timestamp)
    )) for timestamp, y in enumerate([30, 70, 30])]

    for result in results:
        assert result.events[1]["in_prohibit"] is False
        assert result.events[1]["zone_entry"] is False
    assert results[-1].left_count == results[-1].right_count == 1


def test_zone_processing_preserves_source_frame_and_tracking_arrays(
    tracking_packet_factory,
):
    engine = EventEngine(Event(polygon=SQUARE))
    packet = tracking_packet_factory([(4, 50, 50), (9, 10, 50)])
    packet.frame[:] = np.arange(100, dtype=np.uint8)[None, :, None]
    originals = {
        name: getattr(packet, name).copy()
        for name in ("frame", "boxes", "scores", "class_ids", "track_ids")
    }
    for name in originals:
        getattr(packet, name).setflags(write=False)

    result = engine.process(packet)

    for name, before in originals.items():
        np.testing.assert_array_equal(getattr(packet, name), before)
    np.testing.assert_array_equal(result.frame, originals["frame"])
    assert_zone_result(result, 4, inside=True, entry=True)
    assert_zone_result(result, 9, inside=False, entry=False)
