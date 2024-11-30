package com.guiguzi.yolo;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class Label {

    public String ls;
    public int    is;
    public int    ai, bi, ci, di;
    public String af, bf, cf, df;
    
    public static final String float2string(float value) {
        return String.format("%.6f", BigDecimal.valueOf(value).setScale(6, RoundingMode.DOWN).floatValue());
    }
    
}
