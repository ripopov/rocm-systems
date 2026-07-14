#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""CLI leaf test: monitor command."""

import json
import multiprocessing
import os
import time

import common
from cli.base import TestCliBase


class TestMonitor(TestCliBase):
    monitor_args = "--power-usage --temperature --base-board-temps --gpu-board-temps --gfx --mem --encoder --decoder --ecc --vram-usage --pcie"

    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "monitor",
            "Monitor Arguments:",
            "Device Arguments:",
            "Command Modifiers:",
            "Watch Arguments:",
        )
        self.RunCmds(cmds)
        return

    def _get_monitor_metric_data(self, monitor1, monitor2, metric, exclude=False):
        data = []
        for i in range(len(monitor1)):
            data.append(
                {
                    "power_usage": None,
                    "hotspot_temperature": None,
                    "memory_temperature": None,
                    "gfx_clk": None,
                    "gfx": None,
                    "mem": None,
                    "vram_used": None,
                }
            )
            if not exclude:
                data[i]["vram_free"] = None
                data[i]["vram_total"] = None
            # Find the larger of the two amounts
            if metric is not None:
                total_gtt = int(metric["gpu_data"][i]["mem_usage"]["total_gtt"]["value"])
                total_vram = int(metric["gpu_data"][i]["mem_usage"]["total_vram"]["value"])
                if total_gtt > total_vram:
                    vram_key = "gtt"
                else:
                    vram_key = "vram"
            for key in data[i]:
                # monitor1 and monitor2 come from the same source amd-smi monitor
                # Meaning if monitor1 is N/A, then monitor2 will be N/A and by extension
                # the  units for both will always be the same
                if isinstance(monitor1[i][key], str) and monitor1[i][key] == "N/A":
                    unit = "N/A"
                    data1 = 0
                    data2 = 0
                else:
                    unit = monitor1[i][key]["unit"]
                    data1 = int(monitor1[i][key]["value"])
                    if monitor2 is not None:
                        if isinstance(monitor2[i][key], str) and monitor2[i][key] == "N/A":
                            unit = "N/A"
                            data2 = 0
                        else:
                            data2 = int(monitor2[i][key]["value"])
                    else:
                        if key == "power_usage":
                            data2 = int(metric["gpu_data"][i]["power"]["socket_power"]["value"])
                        elif key == "hotspot_temperature":
                            data2 = int(metric["gpu_data"][i]["temperature"]["hotspot"]["value"])
                        elif key == "memory_temperature":
                            data2 = int(metric["gpu_data"][i]["temperature"]["mem"]["value"])
                        elif key == "gfx_clk":
                            data2 = int(metric["gpu_data"][i]["clock"]["gfx_0"]["clk"]["value"])
                        elif key == "gfx":
                            data2 = int(metric["gpu_data"][i]["usage"]["gfx_activity"]["value"])
                        elif key == "mem":
                            data2 = int(metric["gpu_data"][i]["usage"]["mm_activity"]["value"])
                        elif key == "vram_used":
                            data2 = int(
                                metric["gpu_data"][i]["mem_usage"][f"used_{vram_key}"]["value"]
                            )
                        elif key == "vram_free":
                            data2 = int(
                                metric["gpu_data"][i]["mem_usage"][f"free_{vram_key}"]["value"]
                            )
                        elif key == "vram_total":
                            data2 = int(
                                metric["gpu_data"][i]["mem_usage"][f"total_{vram_key}"]["value"]
                            )
                data[i][key] = [data1, data2, abs(data1 - data2), unit]
        return data

    def _compare_monitor_metric_data(self, component, data, data_baseline=None):
        failures = []
        successes = []
        diff_percent = 0.1
        for i in range(len(data)):
            msg_title = f"Monitor to {component}: gpu={i}"
            msg_header = f"{'key':>20s} ({'Unit':>4s}): {'Monitor':>8s} {component:>8s}  {'Diff':>8s}   {'Threshold':>8s} {'Status':>7s}"
            for key in data[i]:
                if data[i][key][3] == "N/A":
                    continue
                # Calculate measurement tolerance
                if data_baseline is not None:
                    max_diff = max(data_baseline[i][key][0], data_baseline[i][key][1])
                else:
                    max_diff = max(data[i][key][0], data[i][key][1])
                max_diff *= diff_percent
                # Allow for some measurement error
                if max_diff < 1.0:
                    max_diff = 1.0

                if data[i][key][2] > max_diff:
                    status = "Failure"
                    compare = ">"
                else:
                    status = "Success"
                    compare = "<"
                _msg = f"{key:>20s} ({data[i][key][3]:>4s}): {data[i][key][0]:>8d} {data[i][key][1]:>8d} ({data[i][key][2]:>8d} {compare} {max_diff:>8.2f}) {status:>7s}"
                if status == "Failure":
                    if len(failures) == 0:
                        failures.append(("", f"Compare {msg_title}"))
                        failures.append(("*" * len(msg_title), msg_header))
                    failures.append((msg_title, _msg))
                else:
                    if len(successes) == 0:
                        successes.append(("", f"Compare {msg_title}"))
                        successes.append(("*" * len(msg_title), msg_header))
                    successes.append((msg_title, _msg))
        return (failures, successes)

    def _print_monitor_results(self, results, fail_on_results=False):
        if results:
            cmd_len = 0
            for cmd, _ in results:
                num = len(cmd)
                if num > cmd_len:
                    cmd_len = num
            cmd_len += 2

            msg = ""
            for cmd, cmd_out in results:
                if cmd:
                    msg += f"\n{self.tab}{cmd:{cmd_len}s} : {cmd_out}"
                else:
                    msg += f"\n{cmd_out}"
            msg = msg.strip()
            msg = f"{self.tab}{msg}"

            # Output to std_out
            if common.verbose == common.VERBOSITY_VERBOSE:
                self.common.print(msg)

            # Output to std_err
            if fail_on_results:
                self.fail(f"Fail:\n\n{msg}")
        return

    def _worker(self, q, start_time: multiprocessing.Value, name: str, cmd: str, timeout: int):
        pid = os.getpid()
        if self.Debug:
            print(f"[{name}] PID={pid} ready, waiting for start time... {start_time.value}")

        # Busy-wait until the shared start time is reached
        while True:
            now = time.perf_counter()
            if now >= start_time.value:
                break
            time.sleep(max(0, start_time.value - now))

        # Record actual start time
        actual_start = time.perf_counter()
        if self.Debug:
            print(
                f"[{name}] Started at {actual_start:.9f} (Δ={actual_start - start_time.value:.9f}s)"
            )

        (rc, std_out, std_err) = self.util.RunCmdSync(cmd, time_out=timeout)

        now = time.perf_counter()
        q.put((name, pid, now, std_out))
        if self.Debug:
            print(f"[{name}] Finished work at {now:.9f}")
        return

    def _multiprocess_commands(self, cmds):
        # Setup queue between processes
        q = multiprocessing.Queue()

        max_timeout = 0
        processes = []
        for name, cmd, time_start_delta, time_out in cmds:
            # Set start time slightly in the future
            future_time = time.perf_counter() + time_start_delta
            # Shared start time (high precision)
            start_time = multiprocessing.Value("d", future_time)
            processes.append(
                multiprocessing.Process(
                    target=self._worker, args=(q, start_time, name, cmd, time_out)
                )
            )
            max_timeout = max(max_timeout, time_out)
            if self.Debug:
                print(
                    f"[Main] Process {name} will start at {future_time:.9f} and time out in {time_out}"
                )

        # Start all processes
        for p in processes:
            p.start()

        # Give processes time to initialize
        time.sleep(1.0)

        # Wait for all to finish
        worker_datas = []
        for p in processes:
            worker_datas.append(q.get(timeout=max_timeout))
            p.join(timeout=max_timeout)

        if self.Debug:
            for worker_data in worker_datas:
                name, pid, time_stamp, data = worker_data
                print(f"name={name} pid={pid} time_stamp={time_stamp}")
            print("[Main] All processes completed.")

        return worker_datas

    def test_monitor_monitor(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor monitor"
        self.common.print(msg)

        # Ensure start time delta is adequate for process to initialize and start
        # and adequate time for time_out to complete process
        time_out = 5.0
        time_start_delta = 2.0
        cmd = f"amd-smi monitor {self.monitor_args} --json"
        cmds = [
            ("monitor1", cmd, time_start_delta, time_out),
            ("monitor2", cmd, time_start_delta, time_out),
        ]
        # Both commands should complete successfully and produce the same results
        worker_datas = self._multiprocess_commands(cmds)

        # Data from monitor1 and monitor2 should be the same
        name, pid, time_stamp, data = worker_datas[0]
        monitor1 = json.loads(data)
        name, pid, time_stamp, data = worker_datas[1]
        monitor2 = json.loads(data)
        data = self._get_monitor_metric_data(monitor1, monitor2, None)
        monitor_failures, monitor_successes = self._compare_monitor_metric_data("Monitor", data)

        # Report results
        self._print_monitor_results(monitor_successes)
        self._print_monitor_results(monitor_failures, fail_on_results=True)
        return

    def test_monitor_metric(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor metric"
        self.common.print(msg)

        # Ensure start time delta is adequate for process to initialize and start
        # and adequate time for time_out to complete process
        time_out = 5.0
        time_start_delta = 2.0
        cmd_monitor = f"amd-smi monitor {self.monitor_args} --json"
        cmd_metric = "amd-smi metric --json"
        cmds = [
            ("monitor", cmd_monitor, time_start_delta, time_out),
            ("metric", cmd_metric, time_start_delta, time_out),
        ]
        # Both commands should complete successfully and produce the same results
        worker_datas = self._multiprocess_commands(cmds)

        # Data from monitor1 and metric1 should be the same
        if worker_datas[0][0] == "metric":
            _, _, time_stamp, data = worker_datas[0]
            metric1 = json.loads(data)
            _, _, time_stamp, data = worker_datas[1]
            monitor1 = json.loads(data)
        else:
            _, _, time_stamp, data = worker_datas[0]
            monitor1 = json.loads(data)
            _, _, time_stamp, data = worker_datas[1]
            metric1 = json.loads(data)
        data = self._get_monitor_metric_data(monitor1, None, metric1)
        monitor_failures, monitor_successes = self._compare_monitor_metric_data("Monitor", data)

        # Report results
        self._print_monitor_results(monitor_successes)
        self._print_monitor_results(monitor_failures, fail_on_results=True)
        return

    def test_monitor_with_workload(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor with workload"
        self.common.print(msg)

        self.rvs_exe = "rvs"
        self.rvs_config_folder = "/opt/rocm/share/rocm-validation-suite/conf"
        # name, options, ramp_time
        # rvs_configs is a list of tuples containing:
        # configuration file name, additional options, the ramp time for the workload
        self.rvs_configs = []
        self.rvs_configs.append(("iet_single.conf", "", 5.0))
        self.rvs_configs.append(("mem.conf", "--numTimes 3", 1.0))

        # Check for workload generator, skip test if not found
        cmd = f"{self.rvs_exe} --version"
        (rc, _, _) = self.util.RunCmdSync(cmd)
        if rc != 0:
            msg = f"{self.tab}rvs not found, skipping test_monitor_with_workload"
            self.common.print(msg)
            self.skipTest(msg)
        # Check for workload generator configuration scripts, skip test if not found
        msgs = []
        for conf_file, _, ramp_time in self.rvs_configs:
            conf_path = f"{self.rvs_config_folder}/{conf_file}"
            if not os.path.exists(conf_path):
                if len(msgs) == 0:
                    msgs.append(f"{self.tab}Skipping test_monitor_with_workload")
                msgs.append(f"\n{self.tab}rvs conf {conf_file} not found")
        if len(msgs) > 0:
            self.common.print(msgs)
            self.skipTest(msgs)

        # Get first baseline results
        cmd_monitor = f"amd-smi monitor {self.monitor_args} --json"
        (rc, data_baseline, std_err) = self.util.RunCmdSync(cmd_monitor)
        if rc != 0:
            msg = f"{self.tab}Monitor with workload test failed, rc={rc}, std_err={std_err}"
            self.common.print(msg)
            self.fail(msg)
        monitor_baseline1 = json.loads(data_baseline)

        # Get second baseline results
        cmd_monitor = f"amd-smi monitor {self.monitor_args} --json"
        (rc, data_baseline, std_err) = self.util.RunCmdSync(cmd_monitor)
        if rc != 0:
            msg = f"{self.tab}Monitor with workload test failed, rc={rc}, std_err={std_err}"
            self.common.print(msg)
            self.fail(msg)
        monitor_baseline2 = json.loads(data_baseline)

        # Get differences between the two baselines
        data_baseline = self._get_monitor_metric_data(
            monitor_baseline1, monitor_baseline2, None, exclude=True
        )

        # Monitor has a delayed start that allows the workload time to initialize and start running.
        # Workload is started immediately and times out after workload has had time to finish
        max_ramp_time = 0.0
        for _, _, ramp_time in self.rvs_configs:
            max_ramp_time = max(max_ramp_time, ramp_time)
        time_out = max_ramp_time + 5.0
        time_start_delta = max_ramp_time + 1.0

        cmds = [("monitor", cmd_monitor, time_start_delta, time_out)]
        time_start_delta = 0.0
        for index, configs in enumerate(self.rvs_configs):
            conf_file, options, ramp_time = configs
            conf_path = f"{self.rvs_config_folder}/{conf_file}"
            cmd_workload = f"{self.rvs_exe} --config {conf_path} {options} --json --quiet"
            cmds.append((f"workload_{index}", cmd_workload, time_start_delta, time_out))
        # Monitor command should complete successfully
        # Worlkload command will timeout and be killed.
        # Workload output is not needed, only the monitor command output is needed for comparison
        worker_datas = self._multiprocess_commands(cmds)

        # Data from monitor_baseline and monitor_workload should not be the same
        monitor_workload = None
        for worker_data in worker_datas:
            if worker_data[0] == "monitor":
                name, pid, time_stamp, data = worker_data
                monitor_workload = json.loads(data)
        if monitor_workload is None:
            msg = f"{self.tab}Monitor with workload test failed, could not get monitor data"
            self.common.print(msg)
            self.fail(msg)

        data = self._get_monitor_metric_data(
            monitor_baseline1, monitor_workload, None, exclude=True
        )
        monitor_failures, monitor_successes = self._compare_monitor_metric_data(
            "Workload", data, data_baseline
        )
        # Results are opposite, want differences in values
        # So switch failure and success criterion
        for index, cmd_data in enumerate(monitor_failures):
            if "Failure" in cmd_data[1]:
                monitor_failures[index] = (
                    cmd_data[0],
                    cmd_data[1].replace("Failure", "Success", 1),
                )
        for index, cmd_data in enumerate(monitor_successes):
            if "Success" in cmd_data[1]:
                monitor_successes[index] = (
                    cmd_data[0],
                    cmd_data[1].replace("Success", "Failure", 1),
                )
        tmp = monitor_failures
        monitor_failures = monitor_successes
        monitor_successes = tmp

        # Report results
        self._print_monitor_results(monitor_successes)
        self._print_monitor_results(monitor_failures, fail_on_results=True)
        return
