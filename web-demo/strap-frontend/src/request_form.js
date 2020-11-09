import React from 'react';
import {
    Badge,
    Container,
    Col,
    Input,
    Form,
    FormText,
    Button,
    Row,
    Dropdown, DropdownToggle, DropdownMenu, DropdownItem,
} from 'reactstrap';
import ReactBootstrapSlider from 'react-bootstrap-slider';

function RequestForm(props) {
    // Build a default text object for the text area

    return (
        <Form>
            <div className="precomputed-div">
                Enter your text or &nbsp;
                <Button color="primary"  onClick={props.transferSentenceRandom}><span>use a random sentence</span></Button>
                &nbsp; or &nbsp;&nbsp;
                <Dropdown isOpen={props.dropdownOpen} toggle={props.toggleExamplesDropDown}>
                    <DropdownToggle color="info" caret>
                        choose an example
                    </DropdownToggle>
                    <DropdownMenu>
                        <DropdownItem tag="a" href="/?id=1adb3684e0d6113205e4e59a">so uh what do you wear to work ---&gt; Shakespeare</DropdownItem>
                        <DropdownItem tag="a" href="/?id=8c5893be42e46daa81a67ae9">yall kissing before marriage? ---&gt; Shakespeare</DropdownItem>
                        <DropdownItem tag="a" href="/?id=781ea338754bb2d09c13adbb">For the foul thief is just at your gate ---&gt; Bible</DropdownItem>
                        <DropdownItem tag="a" href="/?id=42f29b78e2ba53ac125739f1">when ur going thru an episode and cant cope anymore ---&gt; Bible</DropdownItem>
                        <DropdownItem tag="a" href="/?id=8021010d649486f2c5783845">Need you my help? ---&gt; Tweets</DropdownItem>
                        <DropdownItem tag="a" href="/?id=b70b007d8ae712d30b08c35a">O, wilt thou leave me so unsatisfied? ---&gt; Tweets</DropdownItem>
                        <DropdownItem tag="a" href="/?id=b63e1a7831b5fb86ff38e247">Now, don’t tell a big fib! ---&gt; Romantic Poetry</DropdownItem>
                        <DropdownItem tag="a" href="/?id=70a7ec42a34f207143e212ee">O, wilt thou leave me so unsatisfied? ---&gt; Romantic Poetry</DropdownItem>
                        <DropdownItem tag="a" href="/?id=d1a242b743b7afecc8433d62">Just missed that by a hair. ---&gt; Conversational Speech</DropdownItem>
                        <DropdownItem tag="a" href="/?id=83e6b9bdb5c0acb2ee6db538">O, much I fear some ill unlucky thing. ---&gt; Conversational Speech</DropdownItem>
                    </DropdownMenu>
                </Dropdown>
            </div>

            <br />
            <Input type="textarea" name="text" id="strapInputText" rows="2" />
            <FormText>Text is truncated at 50 subwords (using the GPT2 tokenizer). Don't know what to type? Try the "use a random sentence" option or explore samples from our <a href="https://github.com/martiansideofthemoon/style-transfer-paraphrase/tree/master/samples/data_samples">our dataset</a>.</FormText>
            <hr />
            <div className="precomputed-div">
                Transfer sentences to the target style&nbsp;&nbsp;
                <Dropdown class="dropdown-style-menu" isOpen={props.styleDropDownOpen} toggle={props.toggleStyleDropDown}>
                    <DropdownToggle color="info" caret>
                        {props.targetStyle === null ? "choose a target style" : props.targetStyle}
                    </DropdownToggle>
                    <DropdownMenu>
                        <DropdownItem onClick={() => props.toggleStyle("Bible")}>Bible</DropdownItem>
                        <DropdownItem onClick={() => props.toggleStyle("Romantic Poetry")}>Romantic Poetry</DropdownItem>
                        <DropdownItem onClick={() => props.toggleStyle("Shakespeare")}>Shakespeare</DropdownItem>
                        <DropdownItem onClick={() => props.toggleStyle("Conversational Speech")}>Conversational Speech</DropdownItem>
                        <DropdownItem onClick={() => props.toggleStyle("Tweets")}>Tweets</DropdownItem>
                    </DropdownMenu>
                </Dropdown>
            </div>
            <hr />
            paraphraser top-p sampling value = {props.settings.top_p_paraphrase}
            <br />
            <ReactBootstrapSlider
                value={props.settings.top_p_paraphrase}
                change={props.changeSliderParaphrase}
                id="top-p-slider-paraphrase"
                step={0.01}
                max={1.0}
                min={0.0}
                orientation="horizontal"/>

            <FormText>The style transfer models were trained with p = 0.0, but feel free to experiment with this slider if the paraphrases are too close to the input. Increasing the p value results in more diverse paraphrases at the expense of content preservation.
                      Refer to <a href="https://arxiv.org/pdf/1904.09751.pdf">Holtzman et al. 2019</a> for more details.</FormText>
            <hr />
            style transfer top-p sampling value = {props.settings.top_p_style}
            <br />
            <ReactBootstrapSlider
                value={props.settings.top_p_style}
                change={props.changeSliderStyle}
                id="top-p-slider-style"
                step={0.01}
                max={1.0}
                min={0.0}
                orientation="horizontal"/>
            <FormText>Increasing the p value results in more diverse stylistic properties, but at the expense of content preservation. Experiment with this slider to get the desired output, you will get different output samples on each run for larger p values. Some styles seem to benefit from higher p values like 0.6 and 0.9 (see Table 15 in <a href="https://arxiv.org/pdf/2010.05700.pdf#page=25">our paper</a> for more details).</FormText>
            <hr />

            <Button color="primary"  onClick={props.transferSentence}><span>Do style transfer!</span></Button>
        </Form>
    );
}

export default RequestForm;