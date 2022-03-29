import React from 'react';
import $ from 'jquery';
import Button from '@material-ui/core/Button';;
import CircularProgress from '@material-ui/core/CircularProgress';
import { withStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import { PROJECT_TYPES } from '../../constants';


const styles = theme => ({
    button: {
        marginTop: "20px",
        marginLeft: "10px",
        width: "280px"
    },
    container: {
        width: "280px",
        maxWidth: "280px",
        display: "inline-block"
    }
  });

function saveAsFile(text, filename) {
    // Step 1: Create the blob object with the text you received
    const type = 'application/text'; // modify or get it from response
    const blob = new Blob([text], {type});

    // Step 2: Create Blob Object URL for that blob
    const url = URL.createObjectURL(blob);

    // Step 3: Trigger downloading the object using that URL
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click(); // triggering it manually
  }


class DownloadUnlabeledDataButton extends React.Component{

    state = {loading: false}

    downloadLabels = (projectName) => {
        const data = new FormData();
        data.append('project_name', projectName);
        this.setState({ loading: true });

        $.ajax({
            url : '/api/download-unlabeled',
            type : 'POST',
            data : data,
            processData: false,  // tell jQuery not to process the data
            contentType: false,  // tell jQuery not to set contentType,
            success : function(data) {
                saveAsFile(data, `${projectName}.suggested-examples.json`);
                this.setState({ loading: false });
            }.bind(this),
            error: function (error) {
                this.setState({ loading: false });
                alert(error);
            }
        });
    }

    selectButtonLabel(projectType){
        switch(projectType) {
            case PROJECT_TYPES.classification:
                return 'Download labels';
            case PROJECT_TYPES.ner:
                return 'Download Suggested Examples';
            case PROJECT_TYPES.entity_disambiguation:
                return 'Download labels 1';
            default:
                return 'Download labels 2'
          }
    }

    render() {
        const { classes } = this.props;

        if(!this.state.loading){
            return (
                <Button
                    variant="contained"
                    color="primary"
                    onClick={(e) => {this.downloadLabels(this.props.projectName)}}
                    className={classes.button}
                >
                    {this.selectButtonLabel(this.props.projectType)}
                </Button>)
        }
        else{
            return (
                <Container className={classes.container}>
                    <CircularProgress
                        className={classes.button}
                    />
                </Container>
            )
        }
    }
};

export default withStyles(styles)(DownloadUnlabeledDataButton);