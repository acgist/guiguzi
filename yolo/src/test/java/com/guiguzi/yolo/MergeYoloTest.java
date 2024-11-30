package com.guiguzi.yolo;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

public class MergeYoloTest {

    @Test
    public void test() {
        final List<String> list = List.of(
            "D:/download/anquanmao",
            "D:/download/anquandai"
        );
        final String output = "D:/download/target";
        final AtomicInteger baseIndex = new AtomicInteger(0);
        list.forEach(v -> {
            try {
                this.merge(baseIndex, v, output);
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }
    
    private void merge(AtomicInteger baseIndex, String path, String output) throws IOException {
        final AtomicInteger max = new AtomicInteger(0);
        System.out.println(baseIndex);
        Files.list(Paths.get(path)).forEach(v -> {
            try {
                System.out.println(v);
                final StringBuilder builder = new StringBuilder();
                final String[] values = Files.readString(v).split("\n");
                for (final String value : values) {
                    final String[] props = value.split(" ");
                    final Integer classId = Integer.valueOf(props[0]);
                    builder.append(baseIndex.get() + classId).append(value.substring(value.indexOf(' '))).append("\n");
                    max.set(Math.max(classId, max.get()));
                }
                Files.write(Paths.get(output, v.getFileName().toString()), builder.toString().getBytes());
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        baseIndex.addAndGet(max.get() + 1);
    }
    
}
