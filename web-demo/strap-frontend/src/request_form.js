import React from 'react';
import {
    Badge,
    Container,
    Col,
    Input,
    Form,
    FormText,
    Button,
    Row
} from 'reactstrap';
import ReactBootstrapSlider from 'react-bootstrap-slider';

function RequestForm(props) {
    // Build a default text object for the text area

    return (
        <Form>
            paraphraser top-p sampling value = {props.settings.top_p_paraphrase}
            <FormText>The style transfer models were trained with p = 0.0. Increasing the p value results in more diverse paraphrases at the expense of content preservation.
                      Refer to <a href="https://arxiv.org/pdf/1904.09751.pdf">Holtzman et al. 2019</a> for more details.</FormText>
            <ReactBootstrapSlider
                value={props.settings.top_p_paraphrase}
                change={props.changeSliderParaphrase}
                id="top-p-slider-paraphrase"
                step={0.01}
                max={1.0}
                min={0.0}
                orientation="horizontal"/>
            <hr />
            style transfer top-p sampling value = {props.settings.top_p_style}
            <FormText>Increasing the p value results in more diverse stylistic properties, but at the expense of content preservation.</FormText>
            <ReactBootstrapSlider
                value={props.settings.top_p_style}
                change={props.changeSliderStyle}
                id="top-p-slider-style"
                step={0.01}
                max={1.0}
                min={0.0}
                orientation="horizontal"/>
            <hr />
            Enter input sentence
            <FormText>Text is truncated at 50 subwords (using the GPT2 tokenizer).</FormText>
            <Input type="textarea" name="text" id="strapInputText" rows="4" /> <br />
            <Button color="primary"  onClick={props.transferSentence}><span>Transfer</span></Button>
        </Form>
    );
}

export default RequestForm;