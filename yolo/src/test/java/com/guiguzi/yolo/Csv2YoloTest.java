package com.guiguzi.yolo;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * sudo apt install imagemagick
 * ls | xargs -I @ identify -format '%f %wx%h\n' @
 */
public class Csv2YoloTest {
    
    static Map<String, Map<String, Integer>> wh = new HashMap<>();
    
    static String[] classes = { /* "监护袖章(红only)", */ "ground", "safebelt", "offground" };
    // static String[] classes = { "badge", "ground", "safebelt", "offground" };

    @Test
    public void testFloat() {
        System.out.println(String.format("%.4f", 1.123456F));
        System.out.println(String.format("%.4f", 1.123436F));
        System.out.println(String.format("%.4f", BigDecimal.valueOf(1.123456F).setScale(4, RoundingMode.DOWN).floatValue()));
        System.out.println(String.format("%.4f", BigDecimal.valueOf(1.123436F).setScale(4, RoundingMode.DOWN).floatValue()));
    }
    
    @Test
    public void test() throws FileNotFoundException, IOException {
        this.readWh("D:/download/wh.txt");
        this.toYolo("D:/download/3train_rname.csv", "D:/download/labels");
    }
    
    private void readWh(String file) throws FileNotFoundException, IOException {
        try(
            final BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
        ) {
            String line;
            while((line = reader.readLine()) != null) {
                final String[] props = line.split(" ");
                if(props.length < 2) {
                    continue;
                }
                final String wh = props[1].strip();
                final int index = wh.indexOf('x');
                Csv2YoloTest.wh.put(props[0], Map.of(
                    "width",  Integer.valueOf(wh.substring(0, index)),
                    "height", Integer.valueOf(wh.substring(index + 1))
                ));
            }
        }
    }
    
    @SuppressWarnings({ "rawtypes", "unchecked" })
    private void toYolo(String file, String path) throws FileNotFoundException, IOException {
        final ObjectMapper mapper = new ObjectMapper();
        try(
            final BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
        ) {
            String line;
            while((line = reader.readLine()) != null) {
                int index = line.indexOf('/');
                if(index < 0) {
                    continue;
                }
                final String filename = line.substring(index + 1, line.indexOf(',', index));
                index = line.indexOf('{');
                final String json = line.substring(index, line.lastIndexOf('}') + 1).replace("\"\"", "\"");
                final Map map = mapper.readValue(json, Map.class);
                final List<Map> items = (List<Map>) map.get("items");
                final StringBuilder builder = new StringBuilder();
                items.forEach(v -> {
                    final Map meta   = (Map) v.get("meta");
                    final Map labels = (Map) v.get("labels");
                    final int classId = ArrayUtils.indexOf(classes, labels.get("标签"));
                    // 去掉臂章
                    if(classId < 0) {
                        System.err.println("文件类型错误：" + labels);
                        return;
                    }
                    final Float[] geometry = new Float[4];
                    final List list = (List) meta.get("geometry");
                    for (int i = 0; i < list.size(); i++) {
                        geometry[i] = Float.valueOf(list.get(i).toString());
                    }
                    float ws = 1.0F / wh.get(filename).get("width");
                    float hs = 1.0F / wh.get(filename).get("height");
                    float x = (geometry[0] + geometry[2]) / 2.0F;
                    float y = (geometry[1] + geometry[3]) / 2.0F;
                    float w = geometry[2] - geometry[0];
                    float h = geometry[3] - geometry[1];
                    x = x * ws;
                    y = y * hs;
                    w = w * ws;
                    h = h * hs;
                    builder.append(classId).append(" ")
                        .append(Label.float2string(x)).append(" ")
                        .append(Label.float2string(y)).append(" ")
                        .append(Label.float2string(w)).append(" ")
                        .append(Label.float2string(h)).append("\n");
                });
                Files.write(Paths.get(path, filename.substring(0, filename.indexOf('.')) + ".txt"), builder.toString().getBytes());
            }
        }
    }
    
}
