import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Table from '@material-ui/core/Table';
import TableHead from '@material-ui/core/TableHead';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';
import TouchAppIcon from '@material-ui/icons/TouchApp';
import IconButton from '@material-ui/core/IconButton';
import Tooltip from '@material-ui/core/Tooltip';


const useStyles = makeStyles({
    tablecell: {
        fontSize: '80%'
    }
  });

const useStyles2 = makeStyles({
    tablecell: {
        fontSize: '80%',
        fontWeight: 'bold',
        color: '#0089de'
    }
  });

const StyledTableCell = (props) => {
    const classes = useStyles();
    return (
        <TableCell 
            align={props.align}
            className={classes.tablecell}
            component={props.component}
            scope={props.scope}
        >
        {props.content}
        </TableCell>
    )
}

const StyledTableCell2 = (props) => {
    const classes = useStyles2();
    return (
        <TableCell
            align={props.align}
            className={classes.tablecell}
            component={props.component}
            scope={props.scope}
        >
        {props.content}
        </TableCell>
    )
}

const RuleTableHead = (props) => (
    <TableHead>
        <TableRow className={props.classes.table}>
            <StyledTableCell content={"Type"}/>
            <StyledTableCell align="right" content="Total"/>
        </TableRow>
    </TableHead>
)

const ExploreButton = (props) => (
    <Tooltip title="Label">
        <IconButton aria-label="explore-label" onClick={(e) => props.exploreLabelled(e, props.label)} name="explore-label-button">
            <TouchAppIcon name="explore-label-icon"/>
        </IconButton>
    </Tooltip>
)

const RuleTableRow0 = (props) => {
    return (
        <TableRow className={props.classes.table}>
            <StyledTableCell content={props.header}/>
            <StyledTableCell
                align="right"
                content={props.totalManualSpans}
            />
        </TableRow>
)}

const RuleTableRow1 = (props) => {
    return (
        <TableRow className={props.classes.table}>
            <StyledTableCell2 content={props.header}/>
            <StyledTableCell2
                align="right" 
                content={props.totalManualDocs}
            />
        </TableRow>
)}

const RuleTableRow2 = (props) => {
    return (
        <TableRow className={props.classes.table}>
            <StyledTableCell2 content={props.header}/>
            <StyledTableCell2
                align="right"
                content={props.totalManualDocs}
            />
            <StyledTableCell2
                align="right"
                content={props.exploreLabelled && <ExploreButton
                            exploreLabelled={props.exploreLabelled}
                            label={props.label}
                        />}
            />
        </TableRow>
)}

const LabelTableNER = (props) => {
    console.log("Inside document summary", props);
    
    return ( 
        <div>
            <TableContainer component={Paper}>
                <Table aria-label="label table">
                    <RuleTableHead classes={props.classes}/>
                    <TableBody>
                        {props.modelClasses.map((row, index) => { 
                            return (
                                <RuleTableRow0
                                    key={`manual-label-${index}`}
                                    header={`${row.name}`}
                                    totalManualDocs={row.totalManualDocs}
                                    totalManualSpans={row.totalManualSpans}
                                    exploreLabelled={props.exploreLabelled}
                                    label={row.id}
                                    classes={props.classes}
                                />
                            )
                        })}
                        <RuleTableRow1
                            header="Annotated examples"
                            totalManualDocs={props.totalManualDocs}
                            exploreLabelled={props.exploreLabelled}
                            label="empty"
                            classes={props.classes}
                        />
                        <RuleTableRow2
                            header="Examples to annotate"
                            totalManualDocs={props.totalDocuments - props.totalManualDocs}
                            exploreLabelled={props.exploreLabelled}
                            label="none"
                            classes={props.classes}
                        />
                    </TableBody>
                </Table>
            </TableContainer>
        </div>
    )
}

export default LabelTableNER;