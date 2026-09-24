from app.sessions import SessionStore


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_known_id_returns_same_session():
    store = SessionStore(clock=FakeClock())
    first, created = store.get_or_create(None)
    again, created_again = store.get_or_create(first.id)
    assert created is True
    assert created_again is False
    assert again is first


def test_unknown_id_creates_a_fresh_session():
    store = SessionStore(clock=FakeClock())
    session, created = store.get_or_create("not-a-real-id")
    assert created is True
    assert session.id != "not-a-real-id"
    assert len(session.id) >= 40


def test_idle_session_expires_after_ttl():
    clock = FakeClock()
    store = SessionStore(ttl_seconds=10, clock=clock)
    first, _ = store.get_or_create(None)
    clock.t = 11
    second, created = store.get_or_create(first.id)
    assert created is True
    assert second is not first
    assert first.id not in store


def test_least_recently_used_session_is_evicted():
    clock = FakeClock()
    store = SessionStore(max_sessions=2, clock=clock)
    a, _ = store.get_or_create(None)
    clock.t = 1
    b, _ = store.get_or_create(None)
    clock.t = 2
    store.get_or_create(a.id)          # touch a, so b is now least recent
    clock.t = 3
    store.get_or_create(None)          # third session evicts b
    assert a.id in store
    assert b.id not in store
    assert len(store) == 2


def test_reset_model_clears_model_state_only():
    store = SessionStore(clock=FakeClock())
    session, _ = store.get_or_create(None)
    session.dataset_name = "x.csv"
    session.model = object()
    session.new_scores = object()
    session.reasons_cache[("training", 1)] = []
    session.reset_model()
    assert session.model is None
    assert session.new_scores is None
    assert session.reasons_cache == {}
    assert session.dataset_name == "x.csv"
