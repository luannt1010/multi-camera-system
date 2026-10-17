"""Event identities, expiry, and camera isolation without pipeline dependencies."""

import numpy as np
import pytest

from src.event_engine import EventEngine
from src.schemas import Event, EventResults


LINE = [[20, 50], [80, 50]]
POLYGON = [[40, 50], [80, 50], [80, 80], [40, 80]]


def test_empty_packet_emits_no_events_and_preserves_counts(tracking_packet_factory):
    engine = EventEngine(Event(line=LINE, polygon=POLYGON))
    engine.process(tracking_packet_factory([(1, 60, 40)], timestamp=0))
    crossed = engine.process(tracking_packet_factory([(1, 60, 60)], timestamp=1))
    assert (crossed.left_count, crossed.right_count) == (0, 1)

    result = engine.process(tracking_packet_factory(timestamp=2))

    assert result.events == {}
    assert (result.left_count, result.right_count) == (0, 1)


@pytest.mark.parametrize(
    "elapsed,expected_crossing",
    [(5.0, True), (5.001, False)],
    ids=["retained-at-exact-ttl", "forgotten-after-ttl"],
)
def test_identity_expires_only_after_ttl(
    tracking_packet_factory, elapsed, expected_crossing
):
    engine = EventEngine(Event(line=LINE, polygon=POLYGON, state_ttl=5))
    engine.process(tracking_packet_factory([(1, 60, 40)], timestamp=0))

    result = engine.process(
        tracking_packet_factory([(1, 60, 60)], timestamp=elapsed)
    )

    assert result.events[1]["crossed"] is expected_crossing
    assert result.events[1]["direction"] == (
        "left_to_right" if expected_crossing else None
    )
    assert result.right_count == int(expected_crossing)
    assert result.left_count == 0
    assert result.events[1]["zone_entry"] is True


def test_ttl_uses_each_tracks_last_observation(tracking_packet_factory):
    engine = EventEngine(Event(line=LINE, polygon=POLYGON, state_ttl=5))
    engine.process(
        tracking_packet_factory([(1, 60, 40), (2, 60, 40)], timestamp=0)
    )
    engine.process(tracking_packet_factory([(1, 60, 40)], timestamp=4))

    result = engine.process(
        tracking_packet_factory([(1, 60, 60), (2, 60, 60)], timestamp=5.1)
    )

    assert result.events[1]["crossed"] is True
    assert result.events[2]["crossed"] is False
    assert result.events[2]["direction"] is None
    assert (result.left_count, result.right_count) == (0, 1)


def test_empty_packets_expire_all_identity_state_after_ttl(tracking_packet_factory):
    engine = EventEngine(Event(line=LINE, polygon=POLYGON, state_ttl=5))
    engine.process(tracking_packet_factory([(1, 60, 60)], timestamp=0))
    # All four maps must be populated so the cleanup assertions cannot pass
    # merely because one of the optional event rules never ran.
    for state in (
        engine.previous_side, engine.previous_center, engine.inside, engine.last_seen
    ):
        assert set(state) == {1}

    at_ttl = engine.process(tracking_packet_factory(timestamp=5))
    assert at_ttl.events == {}
    assert set(engine.last_seen) == {1}

    after_ttl = engine.process(tracking_packet_factory(timestamp=5.001))

    assert after_ttl.events == {}
    for state in (
        engine.previous_side, engine.previous_center, engine.inside, engine.last_seen
    ):
        assert state == {}


def test_reused_id_after_ttl_has_no_false_crossing_and_can_enter_zone_again(
    tracking_packet_factory,
):
    # The polygon spans both sides of the line: the old and new identity are
    # inside the same zone, but a gap beyond TTL must allow a fresh entry.
    polygon = [[30, 20], [80, 20], [80, 80], [30, 80]]
    engine = EventEngine(Event(line=LINE, polygon=polygon, state_ttl=5))
    first = engine.process(tracking_packet_factory([(7, 60, 40)], timestamp=0))
    assert first.events[7]["zone_entry"] is True

    result = engine.process(tracking_packet_factory([(7, 60, 60)], timestamp=6))

    assert result.events[7] == {
        "in_prohibit": True,
        "zone_entry": True,
        "crossed": False,
        "direction": None,
    }
    assert result.left_count == result.right_count == 0


def test_short_empty_gap_preserves_identity_for_a_real_crossing(
    tracking_packet_factory,
):
    engine = EventEngine(Event(line=LINE, polygon=POLYGON, state_ttl=5))
    engine.process(tracking_packet_factory([(1, 60, 40)], timestamp=0))
    empty = engine.process(tracking_packet_factory(timestamp=2))
    assert empty.events == {}

    result = engine.process(tracking_packet_factory([(1, 60, 60)], timestamp=4))

    assert result.events[1]["crossed"] is True
    assert result.events[1]["direction"] == "left_to_right"
    assert result.events[1]["zone_entry"] is True
    assert (result.left_count, result.right_count) == (0, 1)


def test_reset_forgets_identities_but_preserves_cumulative_counts(
    tracking_packet_factory,
):
    polygon = [[30, 20], [80, 20], [80, 80], [30, 80]]
    engine = EventEngine(Event(line=LINE, polygon=polygon))
    engine.process(tracking_packet_factory([(1, 60, 40)], timestamp=0))
    crossed = engine.process(tracking_packet_factory([(1, 60, 60)], timestamp=1))
    assert crossed.events[1]["crossed"] is True
    assert crossed.events[1]["zone_entry"] is False

    engine.reset()

    assert (engine.left_count, engine.right_count) == (0, 1)
    for state in (
        engine.previous_side, engine.previous_center, engine.inside, engine.last_seen
    ):
        assert state == {}
    result = engine.process(tracking_packet_factory([(1, 60, 40)], timestamp=2))
    assert result.events[1]["crossed"] is False
    assert result.events[1]["direction"] is None
    assert result.events[1]["zone_entry"] is True
    assert (result.left_count, result.right_count) == (0, 1)


def test_camera_engines_isolate_the_same_track_id(tracking_packet_factory):
    camera_a = EventEngine(Event(line=LINE, polygon=POLYGON))
    camera_b = EventEngine(Event(line=LINE, polygon=POLYGON))
    camera_a.process(
        tracking_packet_factory([(1, 60, 40)], timestamp=0, camera_id="cam_a")
    )
    first_b = camera_b.process(
        tracking_packet_factory([(1, 60, 60)], timestamp=0, camera_id="cam_b")
    )
    assert first_b.events[1]["crossed"] is False
    assert first_b.events[1]["zone_entry"] is True

    result_a = camera_a.process(
        tracking_packet_factory([(1, 60, 60)], timestamp=1, camera_id="cam_a")
    )
    result_b = camera_b.process(
        tracking_packet_factory([(1, 60, 40)], timestamp=1, camera_id="cam_b")
    )

    assert result_a.camera_id == "cam_a"
    assert result_b.camera_id == "cam_b"
    assert result_a.events[1]["direction"] == "left_to_right"
    assert result_b.events[1]["direction"] == "right_to_left"
    assert result_a.events[1]["zone_entry"] is True
    assert result_b.events[1]["in_prohibit"] is False
    assert (result_a.left_count, result_a.right_count) == (0, 1)
    assert (result_b.left_count, result_b.right_count) == (1, 0)


def test_one_frame_can_generate_line_crossing_and_zone_entry(
    tracking_packet_factory,
):
    engine = EventEngine(Event(line=LINE, polygon=POLYGON))
    first = engine.process(tracking_packet_factory([(42, 60, 40)], timestamp=0))
    assert first.events[42]["zone_entry"] is False
    assert first.events[42]["crossed"] is False

    result = engine.process(tracking_packet_factory([(42, 60, 60)], timestamp=1))

    assert result.events[42] == {
        "in_prohibit": True,
        "zone_entry": True,
        "crossed": True,
        "direction": "left_to_right",
    }
    assert (result.left_count, result.right_count) == (0, 1)


def test_disabled_rules_return_no_events_and_store_no_identity_state(
    tracking_packet_factory,
):
    engine = EventEngine(Event())
    for timestamp, objects in enumerate(([(1, 60, 40)], [], [(1, 60, 60)])):
        result = engine.process(tracking_packet_factory(objects, timestamp=timestamp))
        assert result.events == {}
        assert result.left_count == result.right_count == 0
    for state in (
        engine.previous_side, engine.previous_center, engine.inside, engine.last_seen
    ):
        assert state == {}


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
def test_results_retain_camera_and_raw_frame_without_mutating_packet(
    tracking_packet_factory, enabled
):
    config = Event(line=LINE, polygon=POLYGON) if enabled else Event()
    engine = EventEngine(config)
    packet = tracking_packet_factory(
        [(1, 60, 60)], timestamp=1, camera_id="entrance_camera"
    )
    packet.frame[:] = np.arange(3, dtype=np.uint8)
    original_frame = packet.frame.copy()
    original_boxes = packet.boxes.copy()
    original_ids = packet.track_ids.copy()

    result = engine.process(packet)

    assert isinstance(result, EventResults)
    assert result.camera_id == "entrance_camera"
    assert result.frame is packet.frame
    np.testing.assert_array_equal(packet.frame, original_frame)
    np.testing.assert_array_equal(packet.boxes, original_boxes)
    np.testing.assert_array_equal(packet.track_ids, original_ids)
