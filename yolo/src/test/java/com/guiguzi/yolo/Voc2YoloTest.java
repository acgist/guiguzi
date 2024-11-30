package com.guiguzi.yolo;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.jupiter.api.Test;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.xml.sax.SAXException;

public class Voc2YoloTest {
    
    static DocumentBuilderFactory factory;
    static DocumentBuilder        builder;
    
    static String[] classes = { "hat", "person" };
    
    static {
        try {
            factory = DocumentBuilderFactory.newInstance();
            builder = factory.newDocumentBuilder();
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
        }
    }
    
    @Test
    public void test() throws IOException {
        final String source = "D:/download/Annotations";
        final String target = "D:/download/labels";
        Files.list(Paths.get(source)).forEach(path -> {
            try {
                final Entry<String, List<Label>> entry = this.readFile(path.toAbsolutePath().toString());
                if(entry.getValue().isEmpty()) {
                    System.out.println("文件没有标记：" + path);
                    return;
                }
                final StringBuilder builder = new StringBuilder();
                entry.getValue().forEach(v -> {
                    builder.append(v.is).append(" ")
                        .append(v.af).append(" ")
                        .append(v.bf).append(" ")
                        .append(v.cf).append(" ")
                        .append(v.df).append("\n");
                });
                Files.write(Paths.get(target, entry.getKey()), builder.toString().getBytes());
            } catch (Exception e) {
                System.err.println("文件解析错误：" + path);
                e.printStackTrace();
            }
        });
    }
    
    private Map.Entry<String, List<Label>> readFile(String file) throws SAXException, IOException {
        final Document document = builder.parse(new File(file));
        final Element root = document.getDocumentElement();
        final int width  = this.getInteger(0, root, "width");
        final int height = this.getInteger(0, root, "height");
        // String filename  = this.getString( 0, root, "filename");
        String filename = file.substring(file.lastIndexOf("\\") + 1);
        filename = filename.substring(0, filename.indexOf('.')) + ".txt";
        final List<Label> list = new ArrayList<>();
        int index = 0;
        while(root.getElementsByTagName("object").getLength() > index) {
            final Label label  = new Label();
            label.ls = this.getString(index, root, "name");
            label.is = ArrayUtils.indexOf(classes, label.ls);
            label.ai = this.getInteger(index, root, "xmin");
            label.bi = this.getInteger(index, root, "xmax");
            label.ci = this.getInteger(index, root, "ymin");
            label.di = this.getInteger(index, root, "ymax");
            float ws = 1.0F / width;
            float hs = 1.0F / height;
            float x = (label.ai + label.bi) / 2.0F;
            float y = (label.ci + label.di) / 2.0F;
            float w = label.bi - label.ai;
            float h = label.di - label.ci;
            x = x * ws;
            y = y * hs;
            w = w * ws;
            h = h * hs;
            label.af = Label.float2string(x);
            label.bf = Label.float2string(y);
            label.cf = Label.float2string(w);
            label.df = Label.float2string(h);
            list.add(label);
            ++index;
        }
        return Map.entry(filename, list);
    }
    
    private String getString(int index, Element root, String name) {
        final Node item = root.getElementsByTagName(name).item(index);
        if(item == null) {
            return null;
        }
        return item.getTextContent();
    }
    
    private Integer getInteger(int index, Element root, String name) {
        final String value = this.getString(index, root, name);
        if(value == null) {
            return null;
        }
        return Integer.valueOf(value);
    }
    
    
/*
<annotation>
        <folder>hat01</folder>
        <filename>000000.jpg</filename>
        <path>D:\dataset\hat01\000000.jpg</path>
        <source>
                <database>Unknown</database>
        </source>
        <size>
                <width>947</width>
                <height>1421</height>
                <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
                <name>hat</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>60</xmin>
                        <ymin>66</ymin>
                        <xmax>910</xmax>
                        <ymax>1108</ymax>
                </bndbox>
        </object>
</annotation>

0 0.512143 0.413089 0.897571 0.733286
 */
    
}
