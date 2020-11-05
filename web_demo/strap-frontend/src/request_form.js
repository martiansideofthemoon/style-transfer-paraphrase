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
            top-p sampling value = {props.settings.top_p}
            <FormText>Increasing the p value results in more diverse questions at the expense of grammaticality since sampling from the tail gets more likely.
                       Refer to <a href="https://arxiv.org/pdf/1904.09751.pdf">Holtzman et al. 2019</a> for more details.</FormText>
            <ReactBootstrapSlider
                value={props.settings.top_p}
                change={props.changeSliderTopP}
                id="top-p-slider"
                step={0.01}
                max={1.0}
                min={0.0}
                orientation="horizontal"/>
            <hr />
            fraction of general questions retained = {props.settings.gen_frac}
            <FormText>Increasing this will retain a larger number of general, high-level overview questions about the
                       input text, which form the upper layer of the QA hierarchy.</FormText>
            <ReactBootstrapSlider
                value={props.settings.gen_frac}
                change={props.changeSliderGenFrac}
                id="general-slider"
                step={0.01}
                max={1.0}
                min={0.0}
                orientation="horizontal"/>
            <hr />
            fraction of specific questions retained = {props.settings.spec_frac}
            <FormText>Increasing this will retain a larger number of specific, low-level drill-down questions about the
                       input text, which form the lower layer of the QA hierarchy.</FormText>
            <ReactBootstrapSlider
                value={props.settings.spec_frac}
                change={props.changeSliderSpecFrac}
                id="specific-slider"
                step={0.01}
                max={1.0}
                min={0.0}
                orientation="horizontal"/>
            <hr />
            Enter input text
            <FormText>Separate paragraphs with newline characters. Do not insert any sensitive information.
                       More than 3 paragraphs / 2000 characters in a paragraph will be truncated.</FormText>
            <Input type="textarea" name="text" id="squashInputText" rows="8" /> <br />
            <Button color="primary"  onClick={props.squashDoc}><span className="squashtitleemph">SQUASH</span></Button>
        </Form>
    );
}

export default RequestForm;