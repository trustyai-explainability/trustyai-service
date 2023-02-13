package org.kie.trustyai.service.data.parsers;

import org.kie.trustyai.explainability.model.Dataframe;
import org.kie.trustyai.service.data.exceptions.DataframeCreateException;

import java.nio.ByteBuffer;

public interface DataParser {

    Dataframe parse(ByteBuffer inputs, ByteBuffer outputs) throws DataframeCreateException;
}
