import io
import unittest

from sim import ui_settings


class RunMarkTest(unittest.TestCase):
    def test_start_line_has_time_and_pid(self):
        buf = io.StringIO()
        ui_settings._mark_run(buf)
        self.assertRegex(buf.getvalue(), r"^=== trainer start \d{4}-\d\d-\d\d \d\d:\d\d:\d\d pid=\d+\n$")


if __name__ == "__main__":
    unittest.main()